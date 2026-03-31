import json
import logging
import subprocess
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import CompositionError

logger = logging.getLogger(__name__)


def run(job_dir: Path, config) -> Path:
    script = GeneratedScript.model_validate(
        json.loads((job_dir / "narration.json").read_text())
    )

    render_dir = job_dir / "scenes" / "render"
    scene_list = job_dir / "scene_list.txt"
    scene_list.write_text("\n".join(
        f"file '{(render_dir / scene.id).with_suffix('.mp4').resolve()}'"
        for scene in script.scenes
    ))

    out_dir = job_dir / "output"
    out_dir.mkdir(exist_ok=True)
    final = out_dir / "final.mp4"

    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(scene_list), "-c", "copy", str(final),
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise CompositionError(f"FFmpeg concat failed: {e.stderr.decode()}")

    logger.info(f"Final video: {final}")
    return final
