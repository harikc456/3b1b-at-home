import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import filelock

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

    tts_cfg = config.tts.kokoro if config.tts.engine == "kokoro" else config.tts.vibevoice
    voice = tts_cfg.voice
    speed = tts_cfg.speed

    segments = [
        (scene.id, seg)
        for scene in script.scenes
        for seg in scene.narration_segments
    ]

    narration_path = job_dir / "narration.json"
    # Write initial narration.json if not present
    if not narration_path.exists():
        narration_path.write_text(script.model_dump_json(indent=2))

    lock = filelock.FileLock(str(narration_path) + ".lock")

    def synth(scene_id: str, seg: NarrationSegment):
        out = audio_dir / seg.id
        try:
            duration = engine.synthesise(seg.text, out, voice=voice, speed=speed)
            mp3 = _wav_to_mp3(out.with_suffix(".wav"))
        except (TTSError, Exception) as e:
            logger.error(f"TTS failed for {seg.id}: {e} — using silence")
            mp3 = _silence(out, seg.estimated_duration)
            duration = seg.estimated_duration
        return seg.id, duration, str(mp3)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(synth, sid, seg): seg for sid, seg in segments}
        for future in as_completed(futures):
            seg_id, duration, mp3_path = future.result()
            with lock:
                data = json.loads(narration_path.read_text())
                for scene in data["scenes"]:
                    for seg in scene["narration_segments"]:
                        if seg["id"] == seg_id:
                            seg["actual_duration"] = duration
                            seg["audio_path"] = mp3_path
                narration_path.write_text(json.dumps(data, indent=2))
            logger.info(f"Synthesised {seg_id} ({duration:.2f}s)")
