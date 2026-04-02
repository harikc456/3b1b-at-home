import json
import logging
import subprocess
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import CompositionError
from mathmotion.utils.ffprobe import measure_duration
from mathmotion.utils.video import freeze_frame

logger = logging.getLogger(__name__)


def _silence(path: Path, duration: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(duration), "-codec:a", "libmp3lame", "-b:a", "192k", str(path),
    ], check=True, capture_output=True)


def _build_audio_track(script: GeneratedScript, job_dir: Path) -> Path:
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    render_dir = job_dir / "scenes" / "render"
    parts = []

    for scene in script.scenes:
        scene_video = render_dir / f"{scene.id}.mp4"
        try:
            scene_video_duration = measure_duration(scene_video)
        except Exception:
            scene_video_duration = sum(
                (seg.actual_duration or 0.0) for seg in scene.narration_segments
            )

        scene_audio_duration = 0.0
        for seg in scene.narration_segments:
            if seg.audio_path and Path(seg.audio_path).exists():
                parts.append(seg.audio_path)
                scene_audio_duration += (seg.actual_duration or 0.0)
            else:
                logger.warning(f"Missing audio for segment {seg.id}")

        # Pad with silence if video is longer than audio for this scene
        padding = scene_video_duration - scene_audio_duration
        if padding > 0.01:
            sil = audio_dir / f"sil_pad_{scene.id}.mp3"
            _silence(sil, padding)
            parts.append(str(sil))

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

    # Build scene list, extending any scene whose video is shorter than its audio
    scene_paths = []
    for scene in script.scenes:
        video_path = render_dir / f"{scene.id}.mp4"
        try:
            video_dur = measure_duration(video_path)
        except Exception:
            video_dur = sum(seg.actual_duration or 0.0 for seg in scene.narration_segments)

        audio_dur = sum(seg.actual_duration or 0.0 for seg in scene.narration_segments)
        gap = audio_dur - video_dur

        if gap > 0.01:
            extended = render_dir / f"{scene.id}_extended.mp4"
            freeze_frame(video_path, gap, extended)
            scene_paths.append(extended)
            logger.info(f"Extended scene {scene.id} by {gap:.2f}s (audio > video)")
        else:
            scene_paths.append(video_path)

    scene_list = job_dir / "scene_list.txt"
    scene_list.write_text("\n".join(
        f"file '{p.resolve()}'" for p in scene_paths
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
