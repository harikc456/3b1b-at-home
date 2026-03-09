import json
import logging
import subprocess
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import CompositionError

logger = logging.getLogger(__name__)


def _silence(path: Path, duration: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(duration), "-codec:a", "libmp3lame", "-b:a", "192k", str(path),
    ], check=True, capture_output=True)


def _build_audio_track(script: GeneratedScript, job_dir: Path) -> Path:
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    parts, current = [], 0.0

    for scene in script.scenes:
        for seg in scene.narration_segments:
            gap = seg.cue_offset - current
            if gap > 0.05:
                sil = audio_dir / f"sil_{len(parts)}.mp3"
                _silence(sil, gap)
                parts.append(str(sil))
            if seg.audio_path:
                parts.append(seg.audio_path)
            current = seg.cue_offset + (seg.actual_duration or 0.0)

    concat = audio_dir / "concat.txt"
    concat.write_text("\n".join(f"file '{Path(p).resolve()}'" for p in parts))
    out = audio_dir / "narration.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat), "-c", "copy", str(out),
    ], check=True, capture_output=True)
    return out


def run(job_dir: Path, config) -> Path:
    script = GeneratedScript.model_validate(
        json.loads((job_dir / "narration.json").read_text())
    )

    audio = _build_audio_track(script, job_dir)

    render_dir = job_dir / "scenes" / "render"
    scene_list = job_dir / "scene_list.txt"
    scene_list.write_text("\n".join(
        f"file '{(render_dir / scene.id).with_suffix('.mp4').resolve()}'" for scene in script.scenes
    ))
    assembled = job_dir / "assembled.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(scene_list), "-c", "copy", str(assembled),
    ], check=True, capture_output=True)

    out_dir = job_dir / "output"
    out_dir.mkdir(exist_ok=True)
    final = out_dir / "final.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(assembled), "-i", str(audio),
            "-c:v", "libx264", "-preset", config.composition.output_preset,
            "-crf", str(config.composition.output_crf),
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            str(final),
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise CompositionError(f"FFmpeg encode failed: {e.stderr.decode()}")

    logger.info(f"Final video: {final}")
    return final
