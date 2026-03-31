import json
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_narration(tmp_path):
    """Write a minimal narration.json with two scenes and return the script dict."""
    data = {
        "title": "T", "topic": "t",
        "scenes": [
            {"id": "scene_1", "class_name": "S1", "manim_code": "",
             "narration_segments": [
                 {"id": "seg_1", "text": "hello", "actual_duration": 2.0,
                  "audio_path": "/tmp/seg1.mp3"}
             ]},
            {"id": "scene_2", "class_name": "S2", "manim_code": "",
             "narration_segments": [
                 {"id": "seg_2", "text": "world", "actual_duration": 1.5,
                  "audio_path": "/tmp/seg2.mp3"}
             ]},
        ],
    }
    (tmp_path / "narration.json").write_text(json.dumps(data))
    render_dir = tmp_path / "scenes" / "render"
    render_dir.mkdir(parents=True)
    (render_dir / "scene_1.mp4").write_bytes(b"fake1")
    (render_dir / "scene_2.mp4").write_bytes(b"fake2")
    return data


def test_compose_calls_ffmpeg_concat(tmp_path):
    """compose.run() calls ffmpeg concat with the two scene files."""
    from mathmotion.stages import compose

    _make_narration(tmp_path)
    cfg = MagicMock()

    with patch("mathmotion.stages.compose.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        compose.run(tmp_path, cfg)

    assert mock_run.call_count == 1
    args = mock_run.call_args[0][0]
    assert "ffmpeg" in args
    assert "-f" in args
    assert "concat" in args


def test_compose_writes_scene_list(tmp_path):
    """compose.run() writes scene_list.txt listing both scene mp4s."""
    from mathmotion.stages import compose

    _make_narration(tmp_path)
    cfg = MagicMock()

    with patch("mathmotion.stages.compose.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        compose.run(tmp_path, cfg)

    scene_list = tmp_path / "scene_list.txt"
    assert scene_list.exists()
    content = scene_list.read_text()
    assert "scene_1.mp4" in content
    assert "scene_2.mp4" in content


def test_compose_returns_final_path(tmp_path):
    """compose.run() returns output/final.mp4."""
    from mathmotion.stages import compose

    _make_narration(tmp_path)
    cfg = MagicMock()

    with patch("mathmotion.stages.compose.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = compose.run(tmp_path, cfg)

    assert result == tmp_path / "output" / "final.mp4"
