# tests/test_api_resume.py
import json
from pathlib import Path

import pytest


def _write_narration(job_dir: Path, with_durations: bool = True):
    seg = {
        "id": "seg_1_0", "text": "hi", "cue_offset": 0.0,
        "actual_duration": 2.0 if with_durations else None,
        "audio_path": "/tmp/x.mp3" if with_durations else None,
    }
    data = {
        "title": "T", "topic": "t",
        "scenes": [{"id": "scene_1", "class_name": "S", "manim_code": "",
                    "narration_segments": [seg]}],
    }
    (job_dir / "narration.json").write_text(json.dumps(data))


def test_preflight_passes_for_tts_start(tmp_path):
    from api.routes import _preflight_validate
    job_dir = tmp_path / "job_1"
    job_dir.mkdir()
    (job_dir / "outline.json").write_text("{}")
    (job_dir / "scene_scripts.json").write_text("{}")
    _write_narration(job_dir, with_durations=False)
    (job_dir / "scenes").mkdir()
    (job_dir / "scenes" / "scene_1.py").write_text("")
    # Should not raise
    _preflight_validate(job_dir, "tts")


def test_preflight_fails_missing_narration_for_tts_start(tmp_path):
    from api.routes import _preflight_validate
    from fastapi import HTTPException
    job_dir = tmp_path / "job_2"
    job_dir.mkdir()
    (job_dir / "outline.json").write_text("{}")
    (job_dir / "scene_scripts.json").write_text("{}")
    # narration.json intentionally missing
    with pytest.raises(HTTPException) as exc_info:
        _preflight_validate(job_dir, "tts")
    assert exc_info.value.status_code == 422
    assert "narration.json" in exc_info.value.detail.lower()


def test_preflight_fails_missing_scene_py_for_tts_start(tmp_path):
    from api.routes import _preflight_validate
    from fastapi import HTTPException
    job_dir = tmp_path / "job_3"
    job_dir.mkdir()
    (job_dir / "outline.json").write_text("{}")
    (job_dir / "scene_scripts.json").write_text("{}")
    _write_narration(job_dir, with_durations=False)
    (job_dir / "scenes").mkdir()
    # scene_1.py intentionally missing
    with pytest.raises(HTTPException) as exc_info:
        _preflight_validate(job_dir, "tts")
    assert exc_info.value.status_code == 422


def test_preflight_fails_missing_duration_for_render_start(tmp_path):
    from api.routes import _preflight_validate
    from fastapi import HTTPException
    job_dir = tmp_path / "job_4"
    job_dir.mkdir()
    (job_dir / "outline.json").write_text("{}")
    (job_dir / "scene_scripts.json").write_text("{}")
    _write_narration(job_dir, with_durations=False)  # no durations!
    (job_dir / "scenes").mkdir()
    (job_dir / "scenes" / "scene_1.py").write_text("")
    with pytest.raises(HTTPException) as exc_info:
        _preflight_validate(job_dir, "render")
    assert exc_info.value.status_code == 422
    assert "actual_duration" in exc_info.value.detail.lower()


def test_preflight_fails_missing_mp4_for_compose_start(tmp_path):
    from api.routes import _preflight_validate
    from fastapi import HTTPException
    job_dir = tmp_path / "job_5"
    job_dir.mkdir()
    (job_dir / "outline.json").write_text("{}")
    (job_dir / "scene_scripts.json").write_text("{}")
    _write_narration(job_dir, with_durations=True)
    (job_dir / "scenes").mkdir()
    (job_dir / "scenes" / "scene_1.py").write_text("")
    render_dir = job_dir / "scenes" / "render"
    render_dir.mkdir()
    # scene_1.mp4 intentionally missing
    with pytest.raises(HTTPException) as exc_info:
        _preflight_validate(job_dir, "compose")
    assert exc_info.value.status_code == 422
    assert "scene_1.mp4" in exc_info.value.detail


def test_preflight_passes_for_outline_start(tmp_path):
    from api.routes import _preflight_validate
    job_dir = tmp_path / "job_6"
    job_dir.mkdir()
    # No files needed for outline (full run)
    _preflight_validate(job_dir, "outline")  # Should not raise


def test_preflight_passes_for_none_start(tmp_path):
    from api.routes import _preflight_validate
    job_dir = tmp_path / "job_7"
    job_dir.mkdir()
    _preflight_validate(job_dir, None)  # Should not raise
