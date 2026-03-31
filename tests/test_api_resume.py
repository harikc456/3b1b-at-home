# tests/test_api_resume.py
import json
from pathlib import Path

import pytest


def _write_narration(job_dir: Path, with_durations: bool = True):
    seg = {
        "id": "seg_1_0", "text": "hi",
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


def test_preflight_passes_for_compose_start_without_durations(tmp_path):
    """compose-only resume does not require actual_duration on segments."""
    from api.routes import _preflight_validate
    job_dir = tmp_path / "job_8"
    job_dir.mkdir()
    (job_dir / "outline.json").write_text("{}")
    (job_dir / "scene_scripts.json").write_text("{}")
    _write_narration(job_dir, with_durations=False)  # no durations — should still pass
    (job_dir / "scenes").mkdir()
    (job_dir / "scenes" / "scene_1.py").write_text("")
    render_dir = job_dir / "scenes" / "render"
    render_dir.mkdir()
    (render_dir / "scene_1.mp4").write_bytes(b"fake")
    # Should not raise — compose does not need actual_duration
    _preflight_validate(job_dir, "compose")


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


def _make_job_state(job_id: str, status: str, created_at: str, topic: str = "test") -> dict:
    return {
        "status": status, "step": "Done", "pct": 100, "output": None,
        "error": None, "topic": topic, "quality": "standard", "level": "undergraduate",
        "voice": None, "tts_engine": "kokoro", "llm_provider": "gemini",
        "last_resumed_from_stage": None, "failed_at_stage": None,
        "created_at": created_at,
    }


def test_get_jobs_returns_list(tmp_path):
    from api.routes import get_jobs

    job_dir = tmp_path / "job_aaa"
    job_dir.mkdir()
    (job_dir / "job_state.json").write_text(
        json.dumps(_make_job_state("job_aaa", "complete", "2026-01-02T00:00:00+00:00"))
    )

    from unittest.mock import patch as p
    with p("api.routes._get_jobs_base_dir", return_value=tmp_path):
        result = get_jobs()

    assert len(result) == 1
    assert result[0]["job_id"] == "job_aaa"
    assert result[0]["status"] == "complete"


def test_get_jobs_applies_running_failed_correction(tmp_path):
    """Jobs with status=running that have no live _jobs entry are returned as failed."""
    from api.routes import get_jobs, _jobs

    job_dir = tmp_path / "job_zzz"
    job_dir.mkdir()
    (job_dir / "job_state.json").write_text(
        json.dumps(_make_job_state("job_zzz", "running", "2026-01-01T00:00:00+00:00"))
    )

    # job_zzz is NOT in _jobs (no live thread)
    _jobs.pop("job_zzz", None)

    from unittest.mock import patch as p
    with p("api.routes._get_jobs_base_dir", return_value=tmp_path):
        result = get_jobs()

    assert result[0]["status"] == "failed"
    assert "restarted" in result[0]["error"].lower()


def test_get_jobs_sorted_by_created_at(tmp_path):
    from api.routes import get_jobs

    for job_id, ts in [("job_old", "2026-01-01T00:00:00+00:00"),
                       ("job_new", "2026-03-01T00:00:00+00:00")]:
        d = tmp_path / job_id
        d.mkdir()
        (d / "job_state.json").write_text(
            json.dumps(_make_job_state(job_id, "complete", ts))
        )

    from unittest.mock import patch as p
    with p("api.routes._get_jobs_base_dir", return_value=tmp_path):
        result = get_jobs()

    assert result[0]["job_id"] == "job_new"
    assert result[1]["job_id"] == "job_old"


def test_get_jobs_caps_at_50(tmp_path):
    from api.routes import get_jobs

    for i in range(60):
        d = tmp_path / f"job_{i:03d}"
        d.mkdir()
        (d / "job_state.json").write_text(
            json.dumps(_make_job_state(f"job_{i:03d}", "complete", f"2026-01-{(i%28)+1:02d}T00:00:00+00:00"))
        )

    from unittest.mock import patch as p
    with p("api.routes._get_jobs_base_dir", return_value=tmp_path):
        result = get_jobs()

    assert len(result) == 50


def _write_job_state(job_dir: Path, status: str = "failed"):
    state = _make_job_state(job_dir.name, status, "2026-01-01T00:00:00+00:00",
                            topic="Fourier transforms")
    (job_dir / "job_state.json").write_text(json.dumps(state))
    return state


def test_resume_returns_job_id(tmp_path):
    from fastapi.testclient import TestClient
    from app import app
    from api.routes import _jobs

    job_dir = tmp_path / "job_aaa"
    job_dir.mkdir()
    _write_job_state(job_dir)
    # Need required files for tts start (skips outline, scene_script, scene_code)
    (job_dir / "outline.json").write_text("{}")
    (job_dir / "scene_scripts.json").write_text("{}")
    _write_narration(job_dir, with_durations=False)
    (job_dir / "scenes").mkdir()
    (job_dir / "scenes" / "scene_1.py").write_text("")

    from unittest.mock import patch as p, MagicMock
    with p("api.routes._get_jobs_base_dir", return_value=tmp_path), \
         p("mathmotion.pipeline.run") as mock_run, \
         p("mathmotion.utils.config.get_config") as mock_cfg:
        mock_cfg.return_value.storage.jobs_dir = str(tmp_path)
        mock_run.return_value = job_dir / "output" / "final.mp4"

        client = TestClient(app)
        resp = client.post("/api/resume/job_aaa",
                           json={"start_from_stage": "tts"})

    assert resp.status_code == 200
    assert resp.json()["job_id"] == "job_aaa"
    # Verify pipeline was called with the correct start_from_stage
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs.get("start_from_stage") == "tts"


def test_resume_returns_404_unknown_job(tmp_path):
    from fastapi.testclient import TestClient
    from app import app
    from unittest.mock import patch as p

    with p("mathmotion.utils.config.get_config") as mock_cfg:
        mock_cfg.return_value.storage.jobs_dir = str(tmp_path)
        client = TestClient(app)
        resp = client.post("/api/resume/job_does_not_exist",
                           json={"start_from_stage": "tts"})

    assert resp.status_code == 404


def test_resume_returns_422_invalid_stage(tmp_path):
    from fastapi.testclient import TestClient
    from app import app
    from unittest.mock import patch as p

    job_dir = tmp_path / "job_bbb"
    job_dir.mkdir()
    _write_job_state(job_dir)

    with p("mathmotion.utils.config.get_config") as mock_cfg:
        mock_cfg.return_value.storage.jobs_dir = str(tmp_path)
        client = TestClient(app)
        resp = client.post("/api/resume/job_bbb",
                           json={"start_from_stage": "not_a_real_stage"})

    assert resp.status_code == 422


def test_resume_returns_409_for_running_job(tmp_path):
    from fastapi.testclient import TestClient
    from app import app
    from api.routes import _jobs
    from unittest.mock import patch as p

    job_dir = tmp_path / "job_ccc"
    job_dir.mkdir()
    _write_job_state(job_dir, status="running")
    # Set up required files for "tts" start (so preflight passes)
    (job_dir / "outline.json").write_text("{}")
    (job_dir / "scene_scripts.json").write_text("{}")
    _write_narration(job_dir, with_durations=False)
    (job_dir / "scenes").mkdir()
    (job_dir / "scenes" / "scene_1.py").write_text("")
    _jobs["job_ccc"] = {"status": "running", "_job_dir": str(job_dir)}

    with p("mathmotion.utils.config.get_config") as mock_cfg:
        mock_cfg.return_value.storage.jobs_dir = str(tmp_path)
        client = TestClient(app)
        resp = client.post("/api/resume/job_ccc",
                           json={"start_from_stage": "tts"})

    assert resp.status_code == 409
    _jobs.pop("job_ccc", None)


def test_status_loads_from_disk_after_restart(tmp_path):
    """Status endpoint lazy-loads job_state.json if job not in memory."""
    from fastapi.testclient import TestClient
    from app import app
    from api.routes import _jobs
    from unittest.mock import patch as p

    job_dir = tmp_path / "job_ddd"
    job_dir.mkdir()
    state = _make_job_state("job_ddd", "complete", "2026-01-01T00:00:00+00:00")
    state["output"] = str(job_dir / "output" / "final.mp4")
    (job_dir / "job_state.json").write_text(json.dumps(state))

    _jobs.pop("job_ddd", None)  # simulate server restart

    with p("mathmotion.utils.config.get_config") as mock_cfg:
        mock_cfg.return_value.storage.jobs_dir = str(tmp_path)
        client = TestClient(app)
        resp = client.get("/api/status/job_ddd")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "complete"


def test_status_corrects_stale_running_on_lazy_load(tmp_path):
    """A stale running job loaded from disk is returned as failed."""
    from fastapi.testclient import TestClient
    from app import app
    from api.routes import _jobs
    from unittest.mock import patch as p

    job_dir = tmp_path / "job_eee"
    job_dir.mkdir()
    state = _make_job_state("job_eee", "running", "2026-01-01T00:00:00+00:00")
    (job_dir / "job_state.json").write_text(json.dumps(state))

    _jobs.pop("job_eee", None)

    with p("mathmotion.utils.config.get_config") as mock_cfg:
        mock_cfg.return_value.storage.jobs_dir = str(tmp_path)
        client = TestClient(app)
        resp = client.get("/api/status/job_eee")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert "restarted" in data["error"].lower()
