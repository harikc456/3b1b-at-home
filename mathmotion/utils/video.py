import subprocess
from pathlib import Path


def freeze_frame(src: Path, duration: float, out: Path) -> Path:
    """Extend *src* video by freezing its last frame for *duration* seconds.

    The result is written to *out* and the path is returned.
    """
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"tpad=stop_mode=clone:stop_duration={duration}",
        "-c:v", "libx264", "-preset", "fast",
        "-an",
        str(out),
    ], check=True, capture_output=True)
    return out
