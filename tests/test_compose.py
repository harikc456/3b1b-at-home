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
