import json
import logging
import re
import subprocess
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import CompositionError

logger = logging.getLogger(__name__)


def compute_drift(actual: float, estimated: float) -> float:
    return abs(actual - estimated) / estimated if estimated else 0.0


def inject_actual_durations(code: str, durations: dict[str, float]) -> str:
    """Replace self.wait() values on lines following # WAIT:{seg_id} comments."""
    lines, out, i = code.split("\n"), [], 0
    while i < len(lines):
        m = re.search(r"# WAIT:(\S+)", lines[i])
        if m and (i + 1) < len(lines) and m.group(1) in durations:
            out.append(lines[i])
            out.append(re.sub(
                r"self\.wait\(.*?\)",
                f"self.wait({durations[m.group(1)]:.3f})",
                lines[i + 1],
            ))
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


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
            current = seg.cue_offset + (seg.actual_duration or seg.estimated_duration)

    concat = audio_dir / "concat.txt"
    concat.write_text("\n".join(f"file '{p}'" for p in parts))
    out = audio_dir / "narration.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat), "-c", "copy", str(out),
    ], check=True, capture_output=True)
    return out


def run(job_dir: Path, config) -> Path:
    # Reload narration — Stage 4 filled in actual_duration and audio_path
    script = GeneratedScript.model_validate(
        json.loads((job_dir / "narration.json").read_text())
    )

    threshold = config.composition.time_stretch_threshold
    strategy = config.composition.sync_strategy

    if strategy == "audio_led":
        # Inject actual TTS durations into scene code and re-render
        durations = {
            seg.id: seg.actual_duration
            for scene in script.scenes
            for seg in scene.narration_segments
            if seg.actual_duration is not None
        }
        scenes_dir = job_dir / "scenes"
        for scene in script.scenes:
            scene_file = scenes_dir / f"{scene.id}.py"
            if scene_file.exists():
                scene_file.write_text(inject_actual_durations(scene_file.read_text(), durations))
        from mathmotion.stages import render
        render.run(script, job_dir, config)

    elif strategy == "time_stretch":
        # Stretch audio segments where drift exceeds threshold
        for scene in script.scenes:
            for seg in scene.narration_segments:
                if not seg.audio_path or not seg.actual_duration:
                    continue
                drift = compute_drift(seg.actual_duration, seg.estimated_duration)
                if drift >= threshold:
                    ratio = max(0.5, min(2.0, seg.actual_duration / seg.estimated_duration))
                    inp = Path(seg.audio_path)
                    out = inp.with_stem(inp.stem + "_stretched")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(inp),
                        "-filter:a", f"atempo={ratio:.4f}", str(out),
                    ], check=True, capture_output=True)
                    seg.audio_path = str(out)
                    seg.actual_duration = seg.estimated_duration

    audio = _build_audio_track(script, job_dir)

    # Concatenate video scenes
    render_dir = job_dir / "scenes" / "render"
    scene_list = job_dir / "scene_list.txt"
    scene_list.write_text("\n".join(
        f"file '{render_dir / scene.id}.mp4'" for scene in script.scenes
    ))
    assembled = job_dir / "assembled.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(scene_list), "-c", "copy", str(assembled),
    ], check=True, capture_output=True)

    # Final encode
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
