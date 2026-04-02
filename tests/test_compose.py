from unittest.mock import patch, MagicMock
from mathmotion.schemas.script import GeneratedScript, Scene, NarrationSegment


def _make_script(scene_id: str, durations: list[float]) -> GeneratedScript:
    segments = [
        NarrationSegment(id=f"{scene_id}_seg_{i}", text="x", actual_duration=d, audio_path="/fake.mp3")
        for i, d in enumerate(durations)
    ]
    return GeneratedScript(
        title="T", topic="T",
        scenes=[Scene(id=scene_id, class_name=f"Scene_{scene_id}", manim_code="", narration_segments=segments)]
    )


def test_compose_extends_video_when_audio_longer(tmp_path):
    """When audio_dur > video_dur for a scene, freeze_frame should be called."""
    script = _make_script("scene_01", [3.0, 4.0])  # audio = 7.0s
    render_dir = tmp_path / "scenes" / "render"
    render_dir.mkdir(parents=True)
    scene_video = render_dir / "scene_01.mp4"
    scene_video.touch()

    (tmp_path / "narration.json").write_text(script.model_dump_json())

    with patch("mathmotion.stages.compose.measure_duration", return_value=5.0), \
         patch("mathmotion.stages.compose.freeze_frame") as mock_freeze, \
         patch("mathmotion.stages.compose._build_audio_track", return_value=tmp_path / "audio.mp3"), \
         patch("mathmotion.stages.compose.subprocess.run"), \
         patch("mathmotion.utils.config.Config"):

        mock_freeze.return_value = render_dir / "scene_01_extended.mp4"

        from mathmotion.stages.compose import run as compose_run
        config = MagicMock()
        config.composition.output_preset = "medium"
        config.composition.output_crf = 23

        try:
            compose_run(tmp_path, config)
        except Exception:
            pass  # final mux may fail; we only care about freeze_frame call

    # audio=7.0, video=5.0 → gap=2.0 → freeze_frame must be called
    mock_freeze.assert_called_once()
    call_args = mock_freeze.call_args[0]
    assert call_args[0] == scene_video       # src
    assert abs(call_args[1] - 2.0) < 0.01   # duration gap
    assert "extended" in str(call_args[2])   # out path


def test_compose_no_extension_when_video_longer(tmp_path):
    """When video_dur >= audio_dur, freeze_frame must NOT be called."""
    script = _make_script("scene_01", [3.0, 2.0])  # audio = 5.0s
    render_dir = tmp_path / "scenes" / "render"
    render_dir.mkdir(parents=True)
    (render_dir / "scene_01.mp4").touch()
    (tmp_path / "narration.json").write_text(script.model_dump_json())

    with patch("mathmotion.stages.compose.measure_duration", return_value=6.0), \
         patch("mathmotion.stages.compose.freeze_frame") as mock_freeze, \
         patch("mathmotion.stages.compose._build_audio_track", return_value=tmp_path / "audio.mp3"), \
         patch("mathmotion.stages.compose.subprocess.run"), \
         patch("mathmotion.utils.config.Config"):

        config = MagicMock()
        config.composition.output_preset = "medium"
        config.composition.output_crf = 23

        from mathmotion.stages.compose import run as compose_run
        try:
            compose_run(tmp_path, config)
        except Exception:
            pass

    mock_freeze.assert_not_called()


import json
import os
from pathlib import Path


def test_render_passes_durations_file_to_subprocess(tmp_path):
    """_render must write a durations JSON and pass MATHMOTION_DURATIONS_FILE env var."""
    from mathmotion.schemas.script import Scene, NarrationSegment
    from mathmotion.stages.render import _render

    scene = Scene(
        id="scene_1",
        class_name="Scene_Test",
        manim_code="from manim import *\nfrom mathmotion.manim_ext import MathMotionScene\nclass Scene_Test(MathMotionScene): pass",
        narration_segments=[
            NarrationSegment(id="seg_0", text="hello world", actual_duration=2.5),
            NarrationSegment(id="seg_1", text="goodbye world", actual_duration=1.3),
        ],
    )

    render_dir = tmp_path / "render"
    render_dir.mkdir()

    captured = {}

    def fake_subprocess_run(cmd, **kwargs):
        env = kwargs.get("env", {})
        dur_path = env.get("MATHMOTION_DURATIONS_FILE", "")
        if dur_path and Path(dur_path).exists():
            captured["durations"] = json.loads(Path(dur_path).read_text())
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    cfg = MagicMock()
    cfg.manim.background_color = "BLACK"
    cfg.manim.timeout_seconds = 60

    with patch("mathmotion.stages.render.subprocess.run", side_effect=fake_subprocess_run):
        try:
            _render(scene, tmp_path, render_dir, "draft", cfg)
        except Exception:
            pass  # No mp4 produced → RenderError is expected; we only check env

    assert "durations" in captured, "MATHMOTION_DURATIONS_FILE was not set or file was missing"
    assert captured["durations"] == [2.5, 1.3]


def test_render_uses_word_count_for_missing_actual_duration(tmp_path):
    """Segments without actual_duration use word-count fallback in durations file."""
    from mathmotion.schemas.script import Scene, NarrationSegment
    from mathmotion.stages.render import _render

    scene = Scene(
        id="scene_1",
        class_name="Scene_Test",
        manim_code="from manim import *\nfrom mathmotion.manim_ext import MathMotionScene\nclass Scene_Test(MathMotionScene): pass",
        narration_segments=[
            NarrationSegment(id="seg_0", text="one two three four five", actual_duration=None),
        ],
    )

    render_dir = tmp_path / "render"
    render_dir.mkdir()

    captured = {}

    def fake_subprocess_run(cmd, **kwargs):
        env = kwargs.get("env", {})
        dur_path = env.get("MATHMOTION_DURATIONS_FILE", "")
        if dur_path and Path(dur_path).exists():
            captured["durations"] = json.loads(Path(dur_path).read_text())
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    cfg = MagicMock()
    cfg.manim.background_color = "BLACK"
    cfg.manim.timeout_seconds = 60

    with patch("mathmotion.stages.render.subprocess.run", side_effect=fake_subprocess_run):
        try:
            _render(scene, tmp_path, render_dir, "draft", cfg)
        except Exception:
            pass

    assert "durations" in captured
    # 5 words / 2.5 = 2.0, floor at 1.0
    assert abs(captured["durations"][0] - 2.0) < 0.01
