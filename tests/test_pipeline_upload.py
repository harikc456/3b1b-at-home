# tests/test_pipeline_upload.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from mathmotion.schemas.script import GeneratedScript, Scene, NarrationSegment


MINIMAL_SCENE_CODE = """from manim import *

class SceneA(Scene):
    def construct(self):
        self.wait(1)
"""

SAMPLE_SCRIPT = GeneratedScript(
    title="Test",
    topic="derivatives",
    scenes=[
        Scene(
            id="scene_a",
            class_name="SceneA",
            manim_code=MINIMAL_SCENE_CODE,
            narration_segments=[
                NarrationSegment(id="seg_1", text="Hello world", cue_offset=0.0)
            ],
        )
    ],
)


def test_pipeline_skips_generate_when_script_provided(tmp_path):
    """When script= is passed, the three generation stages must NOT be called."""
    cfg = MagicMock()
    cfg.manim.default_quality = "draft"
    cfg.storage.jobs_dir = str(tmp_path)
    cfg.llm.repair_max_retries = 0
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"

    with (
        patch("mathmotion.pipeline.get_provider") as mock_provider,
        patch("mathmotion.pipeline.get_engine"),
        patch("mathmotion.pipeline.tts.run"),
        patch("mathmotion.pipeline.inject_actual_durations", side_effect=lambda text, _: text),
        patch("mathmotion.pipeline._run_render_repair_loop", return_value={}),
        patch("mathmotion.pipeline.compose.run", return_value=tmp_path / "out.mp4"),
        patch("mathmotion.pipeline.outline_stage.run") as mock_outline,
        patch("mathmotion.pipeline.scene_script_stage.run") as mock_scripts,
        patch("mathmotion.pipeline.scene_code_stage.run") as mock_code,
    ):
        from mathmotion.pipeline import run as pipeline_run
        pipeline_run(
            topic="derivatives",
            config=cfg,
            script=SAMPLE_SCRIPT,
        )

        mock_outline.assert_not_called()
        mock_scripts.assert_not_called()
        mock_code.assert_not_called()


def test_pipeline_writes_files_when_script_provided(tmp_path):
    """Scene files and narration.json must be written when script= is passed."""
    cfg = MagicMock()
    cfg.manim.default_quality = "draft"
    cfg.storage.jobs_dir = str(tmp_path)
    cfg.llm.repair_max_retries = 0
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"

    def fake_tts(script, job_dir, config, engine):
        import json as _json
        data = _json.loads((job_dir / "narration.json").read_text())
        for scene in data["scenes"]:
            for seg in scene["narration_segments"]:
                seg["actual_duration"] = 1.0
        (job_dir / "narration.json").write_text(_json.dumps(data))

    with (
        patch("mathmotion.pipeline.get_provider"),
        patch("mathmotion.pipeline.get_engine"),
        patch("mathmotion.pipeline.tts.run", side_effect=fake_tts),
        patch("mathmotion.pipeline.inject_actual_durations", side_effect=lambda text, _: text),
        patch("mathmotion.pipeline._run_render_repair_loop", return_value={}),
        patch("mathmotion.pipeline.compose.run", return_value=tmp_path / "out.mp4"),
        patch("mathmotion.pipeline.outline_stage.run"),
        patch("mathmotion.pipeline.scene_script_stage.run"),
        patch("mathmotion.pipeline.scene_code_stage.run"),
    ):
        from mathmotion.pipeline import run as pipeline_run
        pipeline_run(
            topic="derivatives",
            config=cfg,
            script=SAMPLE_SCRIPT,
        )

    job_dirs = list(tmp_path.iterdir())
    assert len(job_dirs) == 1
    job_dir = job_dirs[0]

    assert (job_dir / "narration.json").exists()
    assert (job_dir / "scenes" / "scene_a.py").exists()
    assert "SceneA" in (job_dir / "scenes" / "scene_a.py").read_text()
