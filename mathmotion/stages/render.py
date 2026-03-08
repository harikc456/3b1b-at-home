import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript, Scene
from mathmotion.utils.errors import RenderError

logger = logging.getLogger(__name__)

QUALITY_FLAGS = {
    "draft":    ("-ql", "854,480",   "24"),
    "standard": ("-qm", "1280,720",  "30"),
    "high":     ("-qh", "1920,1080", "60"),
}

FALLBACK_TEMPLATE = '''from manim import *
class Scene_Fallback_{sid}(Scene):
    def construct(self):
        self.play(Write(Text("{title}", font_size=48)))
        self.wait({dur})
'''


def _render(scene: Scene, scenes_dir: Path, render_dir: Path, quality: str, config) -> Path:
    scene_path = scenes_dir / f"{scene.id}.py"
    q_flag, resolution, fps = QUALITY_FLAGS.get(quality, QUALITY_FLAGS["standard"])

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run([
            "manim", "render",
            str(scene_path), scene.class_name,
            "--output_file", f"{scene.id}.mp4",
            "--media_dir", tmp,
            "--resolution", resolution,
            "--frame_rate", fps,
            "--renderer", "cairo",
            "--format", "mp4",
            "--disable_caching",
            "--background_color", config.manim.background_color,
            "--write_to_movie",
            q_flag,
        ], capture_output=True, text=True, timeout=config.manim.timeout_seconds)

        if result.returncode != 0:
            raise RenderError(scene.id, result.stderr)

        mp4s = list(Path(tmp).rglob("*.mp4"))
        if not mp4s:
            raise RenderError(scene.id, "No .mp4 produced by Manim")

        out = render_dir / f"{scene.id}.mp4"
        shutil.copy(mp4s[0], out)
        logger.info(f"Rendered {scene.id} → {out}")
        return out


def _fallback(scene: Scene, scenes_dir: Path, render_dir: Path, config) -> Path:
    code = FALLBACK_TEMPLATE.format(
        sid=scene.id,
        title=scene.id.replace("_", " ").title(),
        dur=max(3, int(scene.estimated_duration)),
    )
    fb_scene = Scene(
        id=scene.id,
        class_name=f"Scene_Fallback_{scene.id}",
        manim_code=code,
        estimated_duration=scene.estimated_duration,
        narration_segments=scene.narration_segments,
    )
    (scenes_dir / f"{scene.id}.py").write_text(code)
    return _render(fb_scene, scenes_dir, render_dir, "draft", config)


def run(script: GeneratedScript, job_dir: Path, config) -> dict[str, Path]:
    quality = config.manim.default_quality
    scenes_dir = job_dir / "scenes"
    render_dir = scenes_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    def do(scene: Scene):
        for attempt in range(3):
            try:
                return scene.id, _render(scene, scenes_dir, render_dir, quality, config)
            except (RenderError, subprocess.TimeoutExpired) as e:
                logger.warning(f"Render attempt {attempt + 1}/3 failed for {scene.id}: {e}")
        logger.error(f"All render attempts failed for {scene.id} — using fallback title card")
        return scene.id, _fallback(scene, scenes_dir, render_dir, config)

    workers = max(1, (os.cpu_count() or 2) // 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(do, script.scenes))
