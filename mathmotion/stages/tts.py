import json
import logging
import subprocess
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript, NarrationSegment
from mathmotion.tts.base import TTSEngine
from mathmotion.utils.errors import TTSError

logger = logging.getLogger(__name__)


def _wav_to_mp3(wav_path: Path) -> Path:
    mp3 = wav_path.with_suffix(".mp3")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", str(mp3),
    ], check=True, capture_output=True)
    return mp3


def _silence(path: Path, duration: float) -> Path:
    mp3 = path.with_suffix(".mp3")
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(duration), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3),
    ], check=True, capture_output=True)
    return mp3


def run(script: GeneratedScript, job_dir: Path, config, engine: TTSEngine) -> None:
    audio_dir = job_dir / "audio" / "segments"
    audio_dir.mkdir(parents=True, exist_ok=True)

    match config.tts.engine:
        case "kokoro":    tts_cfg = config.tts.kokoro
        case "vibevoice": tts_cfg = config.tts.vibevoice
        case n:           raise ValueError(f"Unknown TTS engine: {n!r}")
    voice = tts_cfg.voice
    speed = tts_cfg.speed

    segments = [
        (scene.id, seg)
        for scene in script.scenes
        for seg in scene.narration_segments
    ]
    total_segs = len(segments)
    logger.debug(f"TTS run: engine={config.tts.engine!r}, scenes={len(script.scenes)}, total_segments={total_segs}, voice={voice!r}, speed={speed}")
    logger.info(f"Synthesising {total_segs} audio segment(s) with engine={config.tts.engine!r}")

    def synth(scene_id: str, seg: NarrationSegment):
        logger.info(f"Synthesising {seg.id} ({len(seg.text)} chars)…")
        scene_audio_dir = audio_dir / scene_id
        scene_audio_dir.mkdir(parents=True, exist_ok=True)
        out = scene_audio_dir / seg.id
        fallback_duration = len(seg.text.split()) / 2.5
        try:
            duration = engine.synthesise(seg.text, out, voice=voice, speed=speed)
            wav = out.with_suffix(".wav")
            mp3 = _wav_to_mp3(wav)
            wav.unlink()
        except (TTSError, Exception) as e:
            logger.error(f"TTS failed for {seg.id}: {e} — using silence", exc_info=True)
            mp3 = _silence(out, fallback_duration)
            duration = fallback_duration
        return duration, str(mp3)

    updated = False
    for sid, seg in segments:
        # Idempotency: skip segments already synthesised in a previous run
        if seg.actual_duration is not None:
            logger.info(f"Skipping {seg.id} — already synthesised ({seg.actual_duration:.2f}s)")
            continue
        
        duration, mp3_path = synth(sid, seg)
        seg.actual_duration = duration
        seg.audio_path = mp3_path
        updated = True

    if updated:
        narration_path = job_dir / "narration.json"
        narration_path.write_text(script.model_dump_json(indent=2))
        logger.info(f"Updated {narration_path}")
