# tests/test_job_state.py
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


def _import():
    from api.routes import _save_job_state, _update_job, _state_lock
    return _save_job_state, _update_job, _state_lock


def test_save_job_state_writes_json(tmp_path):
    _save_job_state, _, _ = _import()
    state = {"status": "running", "step": "Testing", "pct": 10}
    _save_job_state("job_abc", state, tmp_path)
    data = json.loads((tmp_path / "job_state.json").read_text())
    assert data["status"] == "running"
    assert data["step"] == "Testing"


def test_save_job_state_is_atomic(tmp_path):
    """Writes via .tmp then renames — no .tmp left behind."""
    _save_job_state, _, _ = _import()
    _save_job_state("job_abc", {"status": "complete"}, tmp_path)
    assert not (tmp_path / "job_state.json.tmp").exists()
    assert (tmp_path / "job_state.json").exists()


def test_save_job_state_swallows_write_error(tmp_path):
    _save_job_state, _, _ = _import()
    # Use a read-only directory to trigger a real write error
    import os
    os.chmod(tmp_path, 0o444)
    try:
        _save_job_state("job_abc", {"status": "running"}, tmp_path)  # should not raise
    finally:
        os.chmod(tmp_path, 0o755)


def test_update_job_merges_fields(tmp_path):
    _save_job_state, _update_job, _ = _import()
    from api.routes import _jobs
    # Seed an entry with a job_dir
    _jobs["job_xyz"] = {"status": "running", "pct": 0, "_job_dir": str(tmp_path)}
    _update_job("job_xyz", pct=50, step="Halfway")
    assert _jobs["job_xyz"]["pct"] == 50
    assert _jobs["job_xyz"]["step"] == "Halfway"
    assert _jobs["job_xyz"]["status"] == "running"  # unchanged
    data = json.loads((tmp_path / "job_state.json").read_text())
    assert data["pct"] == 50


def test_reset_orphaned_jobs_resets_running(tmp_path):
    from api.routes import reset_orphaned_jobs, _jobs

    job_dir = tmp_path / "job_aaa"
    job_dir.mkdir()
    state = {
        "status": "running",
        "step": "Rendering",
        "pct": 50,
        "error": None,
        "topic": "t",
        "quality": "standard",
        "level": "undergraduate",
        "voice": None,
        "tts_engine": "kokoro",
        "llm_provider": "gemini",
        "last_resumed_from_stage": None,
        "failed_at_stage": None,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    (job_dir / "job_state.json").write_text(json.dumps(state))

    reset_orphaned_jobs(jobs_base_dir=tmp_path)

    data = json.loads((job_dir / "job_state.json").read_text())
    assert data["status"] == "failed"
    assert "restarted" in data["error"].lower()


def test_reset_orphaned_jobs_deletes_tmp_files(tmp_path):
    from api.routes import reset_orphaned_jobs

    job_dir = tmp_path / "job_bbb"
    job_dir.mkdir()
    stale_tmp = job_dir / "job_state.json.tmp"
    stale_tmp.write_text("{}")

    reset_orphaned_jobs(jobs_base_dir=tmp_path)

    assert not stale_tmp.exists()


def test_reset_orphaned_jobs_ignores_complete_jobs(tmp_path):
    from api.routes import reset_orphaned_jobs

    job_dir = tmp_path / "job_ccc"
    job_dir.mkdir()
    state = {"status": "complete", "error": None}
    (job_dir / "job_state.json").write_text(json.dumps(state))

    reset_orphaned_jobs(jobs_base_dir=tmp_path)

    data = json.loads((job_dir / "job_state.json").read_text())
    assert data["status"] == "complete"  # unchanged
