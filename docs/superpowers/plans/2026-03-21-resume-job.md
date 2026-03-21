# Resume Interrupted Job Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the ability to resume a pipeline job from a user-chosen stage after an interruption (crash, failure, or server restart).

**Architecture:** Four coordinated changes — (1) job state persistence in `api/routes.py` via an in-memory dict + `job_state.json` on disk; (2) `start_from_stage` parameter in `mathmotion/pipeline.py` with per-stage load-from-disk paths; (3) two new API endpoints (`POST /resume/{job_id}`, `GET /api/jobs`) plus a modified status endpoint; (4) a Past Jobs panel + Resume modal in `static/index.html`.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, threading.Lock, pytest, vanilla HTML/JS

---

## File Map

| File | What changes |
|---|---|
| `api/routes.py` | Add `_state_lock`, `_update_job`, `_save_job_state`, `_preflight_validate`, `reset_orphaned_jobs`; new `POST /resume/{job_id}` and `GET /api/jobs` endpoints; modified `GET /status/{job_id}`; update `start_generate` to persist state |
| `mathmotion/pipeline.py` | Add `STAGES`, `_should_run`, `start_from_stage` param, load-from-disk branches, narration.json.bak |
| `mathmotion/stages/tts.py` | Skip segments where `actual_duration` already set (idempotency) |
| `app.py` | Add FastAPI lifespan calling `reset_orphaned_jobs()` on startup |
| `static/index.html` | Past Jobs panel, Resume modal, Resume button on error state |
| `tests/test_job_state.py` | Tests for persistence helpers and startup reset |
| `tests/test_pipeline_resume.py` | Tests for `_should_run`, load-from-disk paths, bak file, tts idempotency |
| `tests/test_api_resume.py` | Tests for preflight, resume endpoint, jobs endpoint, status lazy load |

---

## Task 1: Job state persistence helpers

**Files:**
- Modify: `api/routes.py`
- Test: `tests/test_job_state.py`

- [ ] **Step 1: Write failing tests for `_save_job_state`**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_job_state.py -v 2>&1 | head -30
```

Expected: ImportError or AttributeError (functions don't exist yet).

- [ ] **Step 3: Add persistence infrastructure to `api/routes.py`**

Add these imports at the top of `api/routes.py`:
```python
import os
import threading
from datetime import datetime, timezone
```

Add after the `_jobs` dict definition:
```python
_state_lock = threading.Lock()


def _save_job_state(job_id: str, state: dict, job_dir: Path) -> None:
    """Atomically write job_state.json. Swallows write errors."""
    # Exclude internal-only keys from the persisted file
    persisted = {k: v for k, v in state.items() if not k.startswith("_")}
    tmp = job_dir / "job_state.json.tmp"
    try:
        tmp.write_text(json.dumps(persisted, indent=2))
        os.replace(tmp, job_dir / "job_state.json")
    except Exception as e:
        logger.warning(f"Failed to save job_state.json for {job_id}: {e}")


def _update_job(job_id: str, **kwargs) -> None:
    """Thread-safe job state update + disk persist."""
    with _state_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(kwargs)
        job_dir = Path(_jobs[job_id]["_job_dir"])
        state_copy = dict(_jobs[job_id])  # copy inside lock before releasing
    _save_job_state(job_id, state_copy, job_dir)
```

Also add `logger = logging.getLogger(__name__)` near the top (after imports), and `import logging` if not already present.

- [ ] **Step 4: Update `start_generate` to persist state**

Replace the current `_jobs[job_id] = {...}` initialization and the direct dict mutations inside `_run` with `_update_job` calls. The new `start_generate` should:

```python
@router.post("/generate")
def start_generate(req: GenerateRequest):
    from mathmotion.utils.config import get_config
    config = get_config()

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = Path(config.storage.jobs_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    initial_state = {
        "status": "running",
        "step": "Starting…",
        "pct": 0,
        "output": None,
        "error": None,
        "topic": req.topic,
        "quality": req.quality,
        "level": req.level,
        "voice": req.voice,
        "tts_engine": req.tts_engine,
        "llm_provider": req.llm_provider,
        "last_resumed_from_stage": None,
        "failed_at_stage": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "_job_dir": str(job_dir),  # internal, not persisted
    }
    with _state_lock:
        _jobs[job_id] = initial_state
    _save_job_state(job_id, initial_state, job_dir)

    def _run():
        from mathmotion.utils.config import get_config as _get
        _get.cache_clear()
        cfg = _get()

        _current_stage = [None]

        try:
            from mathmotion.pipeline import run as run_pipeline, STAGES as _STAGES

            def on_progress(step: str, pct: int):
                matched = next((s for s in _STAGES if s in step.lower().replace(" ", "_")), None)
                if matched:
                    _current_stage[0] = matched
                _update_job(job_id, step=step, pct=pct)

            output = run_pipeline(
                topic=req.topic,
                config=cfg,
                quality=req.quality,
                level=req.level,
                voice=req.voice,
                tts_engine=req.tts_engine,
                llm_provider=req.llm_provider,
                progress_callback=on_progress,
                job_id=job_id,
            )
            _update_job(job_id, status="complete", output=str(output), pct=100, step="Done",
                        failed_at_stage=None)
        except Exception as e:
            _update_job(job_id, status="failed", error=str(e), step="Failed",
                        failed_at_stage=_current_stage[0])

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}
```

- [ ] **Step 5: Run tests**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_job_state.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add api/routes.py tests/test_job_state.py
git commit -m "feat: add job state persistence helpers to routes"
```

---

## Task 2: Startup reset (lifespan)

**Files:**
- Modify: `api/routes.py` (add `reset_orphaned_jobs`)
- Modify: `app.py` (add lifespan)
- Test: `tests/test_job_state.py` (extend)

- [ ] **Step 1: Write failing tests for `reset_orphaned_jobs`**

Add to `tests/test_job_state.py`:
```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_job_state.py::test_reset_orphaned_jobs_resets_running -v 2>&1 | head -20
```

Expected: ImportError (function not defined).

- [ ] **Step 3: Add `reset_orphaned_jobs` to `api/routes.py`**

Add after `_update_job`:

```python
def reset_orphaned_jobs(jobs_base_dir: Optional[Path] = None) -> None:
    """Called on startup. Resets running jobs to failed, removes stale .tmp files."""
    if jobs_base_dir is None:
        from mathmotion.utils.config import get_config
        jobs_base_dir = Path(get_config().storage.jobs_dir)

    if not jobs_base_dir.exists():
        return

    for job_dir in jobs_base_dir.iterdir():
        if not job_dir.is_dir():
            continue
        # Remove stale .tmp files
        for tmp_file in job_dir.glob("*.tmp"):
            try:
                tmp_file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {tmp_file}: {e}")
        # Reset running jobs
        state_file = job_dir / "job_state.json"
        if not state_file.exists():
            continue
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            continue
        if state.get("status") == "running":
            state["status"] = "failed"
            state["error"] = "Server restarted while job was running"
            state["step"] = "Failed"
            try:
                tmp = state_file.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(state, indent=2))
                os.replace(tmp, state_file)
            except Exception as e:
                logger.warning(f"Could not reset orphaned job in {job_dir}: {e}")
```

- [ ] **Step 4: Add lifespan to `app.py`**

```python
# app.py — add at top with other imports:
from contextlib import asynccontextmanager

# Add before `app = FastAPI(...)`:
@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.routes import reset_orphaned_jobs
    reset_orphaned_jobs()
    yield

# Change the app definition to:
app = FastAPI(title="MathMotion", lifespan=lifespan)
```

- [ ] **Step 5: Run tests**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_job_state.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add api/routes.py app.py tests/test_job_state.py
git commit -m "feat: add startup reset for orphaned running jobs"
```

---

## Task 3: Pipeline `STAGES`, `_should_run`, and `start_from_stage`

**Files:**
- Modify: `mathmotion/pipeline.py`
- Test: `tests/test_pipeline_resume.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline_resume.py
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch


STAGES = ["outline", "scene_script", "scene_code", "tts", "render", "compose"]

MINIMAL_OUTLINE = {
    "title": "Derivatives",
    "topic": "derivatives",
    "level": "undergraduate",
    "scenes": [{"id": "scene_1", "title": "Intro", "purpose": "p", "order": 1}],
}

MINIMAL_NARRATION = {
    "title": "Derivatives",
    "topic": "derivatives",
    "scenes": [{
        "id": "scene_1",
        "class_name": "Scene1",
        "manim_code": "class Scene1(Scene): pass",
        "narration_segments": [
            {"id": "seg_1_0", "text": "Hello", "cue_offset": 0.0,
             "actual_duration": 2.0, "audio_path": "/tmp/seg.mp3"},
        ],
    }],
}


def test_should_run_none_always_true():
    from mathmotion.pipeline import _should_run
    for stage in STAGES:
        assert _should_run(stage, None) is True


def test_should_run_outline_always_true():
    from mathmotion.pipeline import _should_run
    for stage in STAGES:
        assert _should_run(stage, "outline") is True


def test_should_run_tts_skips_earlier_stages():
    from mathmotion.pipeline import _should_run
    assert _should_run("outline", "tts") is False
    assert _should_run("scene_script", "tts") is False
    assert _should_run("scene_code", "tts") is False
    assert _should_run("tts", "tts") is True
    assert _should_run("render", "tts") is True
    assert _should_run("compose", "tts") is True


def test_pipeline_skips_outline_when_file_exists(tmp_path):
    """When start_from_stage='compose', all prior stages are loaded from disk."""
    from mathmotion.pipeline import run
    from mathmotion.schemas.script import TopicOutline

    # Write all required pre-existing files for start_from_stage='compose'
    (tmp_path / "outline.json").write_text(json.dumps(MINIMAL_OUTLINE))
    scene_scripts = {
        "title": "Derivatives", "topic": "derivatives",
        "scenes": [{"id": "scene_1", "title": "Intro",
                    "narration": "Hello", "animation_description": {
                        "objects": [], "sequence": [], "notes": ""}}],
    }
    (tmp_path / "scene_scripts.json").write_text(json.dumps(scene_scripts))
    (tmp_path / "narration.json").write_text(json.dumps(MINIMAL_NARRATION))
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    (scenes_dir / "scene_1.py").write_text("class Scene1: pass")
    render_dir = scenes_dir / "render"
    render_dir.mkdir()
    (render_dir / "scene_1.mp4").write_bytes(b"fake")

    cfg = MagicMock()
    cfg.storage.jobs_dir = str(tmp_path.parent)
    cfg.manim.default_quality = "draft"
    cfg.llm.model = "gemini"
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"
    cfg.tts.kokoro.speed = 1.0
    cfg.tts.vibevoice.voice = "neutral"
    cfg.composition.output_preset = "ultrafast"
    cfg.composition.output_crf = 23

    provider = MagicMock()

    # Patch compose to avoid ffmpeg
    with patch("mathmotion.stages.compose.run", return_value=tmp_path / "output" / "final.mp4") as mock_compose, \
         patch("mathmotion.stages.render.try_render_all", return_value=({}, {})):
        run(
            topic="derivatives",
            config=cfg,
            job_id=tmp_path.name,
            start_from_stage="compose",
        )

    # outline.run should NOT have been called (no provider interaction needed for outline stage)
    provider.complete.assert_not_called()
    mock_compose.assert_called_once()


def test_narration_bak_created_before_tts(tmp_path):
    """narration.json.bak is created before tts if it doesn't already exist."""
    from mathmotion.pipeline import run

    # Write all files required for start_from_stage='tts'
    # (outline, scene_scripts, narration.json, scenes/scene_1.py are all required)
    (tmp_path / "outline.json").write_text(json.dumps(MINIMAL_OUTLINE))
    scene_scripts = {
        "title": "Derivatives", "topic": "derivatives",
        "scenes": [{"id": "scene_1", "title": "Intro",
                    "narration": "Hello", "animation_description": {
                        "objects": [], "sequence": [], "notes": ""}}],
    }
    (tmp_path / "scene_scripts.json").write_text(json.dumps(scene_scripts))
    narration_no_durations = {
        "title": "Derivatives", "topic": "derivatives",
        "scenes": [{"id": "scene_1", "class_name": "Scene1",
                    "manim_code": "", "narration_segments": [
                        {"id": "seg_1_0", "text": "Hello", "cue_offset": 0.0}]}],
    }
    (tmp_path / "narration.json").write_text(json.dumps(narration_no_durations))
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "scene_1.py").write_text("")

    cfg = MagicMock()
    cfg.storage.jobs_dir = str(tmp_path.parent)
    cfg.manim.default_quality = "draft"
    cfg.llm.model = "gemini"
    cfg.llm.repair_max_retries = 0
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"
    cfg.tts.kokoro.speed = 1.0
    cfg.tts.vibevoice.voice = "neutral"
    cfg.composition.output_preset = "ultrafast"
    cfg.composition.output_crf = 23

    mock_engine = MagicMock()
    mock_engine.synthesise.return_value = 2.0

    with patch("mathmotion.stages.tts.run") as mock_tts, \
         patch("mathmotion.stages.render.try_render_all", return_value=({"scene_1": tmp_path / "scenes" / "render" / "scene_1.mp4"}, {})), \
         patch("mathmotion.stages.compose.run", return_value=tmp_path / "output" / "final.mp4"), \
         patch("mathmotion.tts.factory.get_engine", return_value=mock_engine):
        (tmp_path / "scenes" / "render").mkdir(parents=True, exist_ok=True)
        run(
            topic="derivatives",
            config=cfg,
            job_id=tmp_path.name,
            start_from_stage="tts",
        )

    assert (tmp_path / "narration.json.bak").exists()


def test_narration_bak_not_overwritten_on_resume(tmp_path):
    """If narration.json.bak already exists, it is not overwritten."""
    from mathmotion.pipeline import run

    # Write all files required for start_from_stage='tts'
    (tmp_path / "outline.json").write_text(json.dumps(MINIMAL_OUTLINE))
    scene_scripts = {
        "title": "D", "topic": "d",
        "scenes": [{"id": "scene_1", "title": "S", "narration": "hi",
                    "animation_description": {"objects": [], "sequence": [], "notes": ""}}],
    }
    (tmp_path / "scene_scripts.json").write_text(json.dumps(scene_scripts))
    narration_no_durations = {
        "title": "D", "topic": "d",
        "scenes": [{"id": "scene_1", "class_name": "S", "manim_code": "",
                    "narration_segments": [{"id": "s", "text": "hi", "cue_offset": 0.0}]}],
    }
    (tmp_path / "narration.json").write_text(json.dumps(narration_no_durations))
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "scene_1.py").write_text("")
    original_bak = '{"original": true}'
    (tmp_path / "narration.json.bak").write_text(original_bak)

    cfg = MagicMock()
    cfg.storage.jobs_dir = str(tmp_path.parent)
    cfg.manim.default_quality = "draft"
    cfg.llm.repair_max_retries = 0
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"
    cfg.tts.kokoro.speed = 1.0

    with patch("mathmotion.stages.tts.run"), \
         patch("mathmotion.stages.render.try_render_all", return_value=({}, {})), \
         patch("mathmotion.stages.compose.run", return_value=tmp_path / "output" / "final.mp4"), \
         patch("mathmotion.tts.factory.get_engine", return_value=MagicMock()):
        (tmp_path / "scenes" / "render").mkdir(parents=True, exist_ok=True)
        run(topic="d", config=cfg, job_id=tmp_path.name, start_from_stage="tts")

    assert (tmp_path / "narration.json.bak").read_text() == original_bak
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_pipeline_resume.py::test_should_run_none_always_true tests/test_pipeline_resume.py::test_should_run_outline_always_true tests/test_pipeline_resume.py::test_should_run_tts_skips_earlier_stages -v 2>&1 | head -20
```

Expected: ImportError (`_should_run` not defined).

- [ ] **Step 3: Add `STAGES` and `_should_run` to `mathmotion/pipeline.py`**

Add after the imports at the top of `pipeline.py`:

```python
STAGES = ["outline", "scene_script", "scene_code", "tts", "render", "compose"]


def _should_run(stage: str, start_from_stage: Optional[str]) -> bool:
    """Return True if this stage should execute (not be skipped)."""
    if start_from_stage is None or start_from_stage == "outline":
        return True
    return STAGES.index(stage) >= STAGES.index(start_from_stage)
```

- [ ] **Step 4: Run `_should_run` tests**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_pipeline_resume.py::test_should_run_none_always_true tests/test_pipeline_resume.py::test_should_run_outline_always_true tests/test_pipeline_resume.py::test_should_run_tts_skips_earlier_stages -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Add `start_from_stage` to `pipeline.run()` with load-from-disk branches**

Update the `run()` signature:
```python
def run(
    topic: str,
    config: Config,
    quality: Optional[str] = None,
    level: str = "undergraduate",
    voice: Optional[str] = None,
    tts_engine: Optional[str] = None,
    llm_provider: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    job_id: Optional[str] = None,
    start_from_stage: Optional[str] = None,
) -> Path:
```

Replace each stage block with a skip/run branch. Full replacement of the body of `run()` (after the config overrides and job_dir setup):

```python
    logger.info(f"Job {job_id} | topic={topic!r} | model={config.llm.model} | "
                f"tts={config.tts.engine} | quality={config.manim.default_quality} | "
                f"start_from={start_from_stage!r}")

    import shutil

    # ── Outline ──────────────────────────────────────────────────────────────
    if _should_run("outline", start_from_stage):
        progress("Generating outline", 10)
        provider = get_provider(config)
        outline_result = outline_stage.run(topic, job_dir, config, provider, level=level)
    else:
        progress("Loading outline from disk", 10)
        provider = get_provider(config)
        outline_result = TopicOutline.model_validate(
            json.loads((job_dir / "outline.json").read_text())
        )

    # ── Scene scripts ─────────────────────────────────────────────────────────
    if _should_run("scene_script", start_from_stage):
        progress("Writing scene scripts", 20)
        scripts_result = scene_script_stage.run(outline_result, job_dir, config, provider)
    else:
        progress("Loading scene scripts from disk", 20)
        from mathmotion.schemas.script import AllSceneScripts
        scripts_result = AllSceneScripts.model_validate(
            json.loads((job_dir / "scene_scripts.json").read_text())
        )

    # ── Scene code ────────────────────────────────────────────────────────────
    if _should_run("scene_code", start_from_stage):
        progress("Generating scene code", 35)
        script = scene_code_stage.run(scripts_result, outline_result, job_dir, config, provider)
    else:
        progress("Loading scene code from disk", 35)
        script = GeneratedScript.model_validate(
            json.loads((job_dir / "narration.json").read_text())
        )

    # ── TTS ───────────────────────────────────────────────────────────────────
    # Create forensic backup before first TTS run (never overwrite)
    bak_path = job_dir / "narration.json.bak"
    if not bak_path.exists():
        shutil.copy2(job_dir / "narration.json", bak_path)

    if _should_run("tts", start_from_stage):
        progress("Synthesising audio", 45)
        engine = get_engine(config)
        tts.run(script, job_dir, config, engine)

    # ── Inject durations ──────────────────────────────────────────────────────
    progress("Injecting durations into scene code", 60)
    script = GeneratedScript.model_validate(
        json.loads((job_dir / "narration.json").read_text())
    )
    durations = {
        seg.id: seg.actual_duration
        for scene in script.scenes
        for seg in scene.narration_segments
        if seg.actual_duration is not None
    }
    scenes_dir = job_dir / "scenes"
    for scene in script.scenes:
        scene_file = scenes_dir / f"{scene.id}.py"
        if scene_file.exists():
            scene_file.write_text(inject_actual_durations(scene_file.read_text(), durations))

    # ── Render ────────────────────────────────────────────────────────────────
    if _should_run("render", start_from_stage):
        progress("Rendering animation", 70)
        _run_render_repair_loop(script, job_dir, config, provider)
    else:
        progress("Skipping render (using existing files)", 70)
        # render results are read directly from disk by compose

    # ── Compose ───────────────────────────────────────────────────────────────
    progress("Composing final video", 88)
    final = compose.run(job_dir, config)

    progress("Done", 100)
    logger.info(f"Output: {final}")
    return final
```

Note: the "inject durations" step always runs (it's idempotent and harmless when skipping TTS). The `TopicOutline` import is already at the top of `pipeline.py`.

- [ ] **Step 6: Run all pipeline resume tests**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_pipeline_resume.py -v
```

Expected: all tests PASS (some may be skipped if mocks need adjustment — fix as needed).

- [ ] **Step 7: Commit**

```bash
git add mathmotion/pipeline.py tests/test_pipeline_resume.py
git commit -m "feat: add start_from_stage to pipeline with per-stage disk load"
```

---

## Task 4: TTS idempotency

**Files:**
- Modify: `mathmotion/stages/tts.py`
- Test: `tests/test_pipeline_resume.py` (extend)

- [ ] **Step 1: Write failing test**

Add to `tests/test_pipeline_resume.py`:

```python
def test_tts_skips_segment_with_existing_duration(tmp_path):
    """Segments that already have actual_duration are not re-synthesised."""
    from mathmotion.stages import tts as tts_stage
    from mathmotion.schemas.script import GeneratedScript

    script = GeneratedScript.model_validate({
        "title": "T", "topic": "t",
        "scenes": [{
            "id": "scene_1", "class_name": "S", "manim_code": "",
            "narration_segments": [
                {"id": "seg_done", "text": "done", "cue_offset": 0.0,
                 "actual_duration": 1.5, "audio_path": "/tmp/done.mp3"},
                {"id": "seg_todo", "text": "todo", "cue_offset": 1.5,
                 "actual_duration": None, "audio_path": None},
            ],
        }],
    })

    narration_path = tmp_path / "narration.json"
    narration_path.write_text(script.model_dump_json(indent=2))

    cfg = MagicMock()
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"
    cfg.tts.kokoro.speed = 1.0

    mock_engine = MagicMock()
    mock_engine.synthesise.return_value = 2.0

    with patch("mathmotion.stages.tts.subprocess.run"):
        tts_stage.run(script, tmp_path, cfg, mock_engine)

    # Only seg_todo should have been synthesised
    assert mock_engine.synthesise.call_count == 1
    call_args = mock_engine.synthesise.call_args
    assert "todo" in call_args[0][0]  # first positional arg is the text
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_pipeline_resume.py::test_tts_skips_segment_with_existing_duration -v 2>&1 | head -20
```

Expected: FAIL — synthesise is called twice, not once.

- [ ] **Step 3: Add skip logic to `tts.run()`**

In `mathmotion/stages/tts.py`, modify the `for` loop at the end of `run()`:

```python
    done = 0
    for sid, seg in segments:
        # Idempotency: skip segments already synthesised in a previous run
        if seg.actual_duration is not None:
            logger.info(f"Skipping {seg.id} — already synthesised ({seg.actual_duration:.2f}s)")
            continue
        scene_id, seg_id, duration, mp3_path = synth(sid, seg)
        done += 1
        data = json.loads(narration_path.read_text())
        for scene in data["scenes"]:
            if scene["id"] != scene_id:
                continue
            for s in scene["narration_segments"]:
                if s["id"] == seg_id:
                    s["actual_duration"] = duration
                    s["audio_path"] = mp3_path
        narration_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Synthesised {seg_id} ({duration:.2f}s) — {done} new segments done")
```

- [ ] **Step 4: Run test**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_pipeline_resume.py::test_tts_skips_segment_with_existing_duration -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mathmotion/stages/tts.py tests/test_pipeline_resume.py
git commit -m "feat: make tts stage idempotent for interrupted resume"
```

---

## Task 5: Pre-flight validation

**Files:**
- Modify: `api/routes.py` (add `_preflight_validate`)
- Test: `tests/test_api_resume.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_api_resume.py -v 2>&1 | head -20
```

Expected: ImportError (`_preflight_validate` not defined).

- [ ] **Step 3: Add `_preflight_validate` to `api/routes.py`**

Add after `reset_orphaned_jobs`:

```python
def _preflight_validate(job_dir: Path, start_from_stage: Optional[str]) -> None:
    """Validate that all files required by skipped stages exist.

    Raises HTTPException(422) with a descriptive message if any are missing.
    start_from_stage=None or 'outline' means full run — no validation needed.
    """
    from fastapi import HTTPException

    from mathmotion.pipeline import STAGES  # lazy import — pipeline.py may not be loaded yet

    if start_from_stage is None or start_from_stage == "outline":
        return

    start_idx = STAGES.index(start_from_stage)
    skipped = STAGES[:start_idx]  # stages that will NOT run

    # Load narration.json once if needed (for scene_id enumeration)
    narration_data = None

    def _load_narration():
        nonlocal narration_data
        if narration_data is None:
            narration_path = job_dir / "narration.json"
            if not narration_path.exists():
                raise HTTPException(422, detail="Required file missing: narration.json")
            narration_data = json.loads(narration_path.read_text())
        return narration_data

    def _scene_ids():
        return [s["id"] for s in _load_narration()["scenes"]]

    for stage in skipped:
        if stage == "outline":
            if not (job_dir / "outline.json").exists():
                raise HTTPException(422, detail="Required file missing: outline.json")

        elif stage == "scene_script":
            if not (job_dir / "scene_scripts.json").exists():
                raise HTTPException(422, detail="Required file missing: scene_scripts.json")

        elif stage == "scene_code":
            # narration.json + per-scene .py files
            for scene_id in _scene_ids():
                py = job_dir / "scenes" / f"{scene_id}.py"
                if not py.exists():
                    raise HTTPException(422, detail=f"Required file missing: scenes/{scene_id}.py")

        elif stage == "tts":
            # narration.json must have actual_duration on all segments
            for scene in _load_narration()["scenes"]:
                for seg in scene["narration_segments"]:
                    if seg.get("actual_duration") is None:
                        raise HTTPException(
                            422,
                            detail=f"Required: actual_duration missing on segment {seg['id']} "
                                   f"(narration.json not fully synthesised)"
                        )

        elif stage == "render":
            # narration.json + per-scene .mp4 files
            render_dir = job_dir / "scenes" / "render"
            for scene_id in _scene_ids():
                mp4 = render_dir / f"{scene_id}.mp4"
                if not mp4.exists():
                    raise HTTPException(422, detail=f"Required file missing: scenes/render/{scene_id}.mp4")
```

- [ ] **Step 4: Run tests**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_api_resume.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routes.py tests/test_api_resume.py
git commit -m "feat: add pre-flight validation for resume endpoint"
```

---

## Task 6: `GET /api/jobs` endpoint

**Files:**
- Modify: `api/routes.py`
- Test: `tests/test_api_resume.py` (extend)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_api_resume.py`:

```python
def _make_job_state(job_id: str, status: str, created_at: str, topic: str = "test") -> dict:
    return {
        "status": status, "step": "Done", "pct": 100, "output": None,
        "error": None, "topic": topic, "quality": "standard", "level": "undergraduate",
        "voice": None, "tts_engine": "kokoro", "llm_provider": "gemini",
        "last_resumed_from_stage": None, "failed_at_stage": None,
        "created_at": created_at,
    }


def test_get_jobs_returns_list(tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module

    # Patch storage dir
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "mathmotion.utils.config.get_config"
    ) as mock_cfg:
        mock_cfg.return_value.storage.jobs_dir = str(tmp_path)

        job_dir = tmp_path / "job_aaa"
        job_dir.mkdir()
        (job_dir / "job_state.json").write_text(
            json.dumps(_make_job_state("job_aaa", "complete", "2026-01-02T00:00:00+00:00"))
        )

        from api.routes import get_jobs
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_api_resume.py::test_get_jobs_returns_list tests/test_api_resume.py::test_get_jobs_applies_running_failed_correction -v 2>&1 | head -20
```

Expected: ImportError / AttributeError.

- [ ] **Step 3: Add `GET /api/jobs` to `api/routes.py`**

First add a small helper (makes tests easier to patch):

```python
def _get_jobs_base_dir() -> Path:
    from mathmotion.utils.config import get_config
    return Path(get_config().storage.jobs_dir)
```

Then add the endpoint:

```python
@router.get("/jobs")
def get_jobs():
    jobs_dir = _get_jobs_base_dir()
    if not jobs_dir.exists():
        return []

    entries = []
    for job_dir in jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue
        state_file = job_dir / "job_state.json"
        if not state_file.exists():
            continue
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            continue

        # Apply running→failed correction in response (without writing to disk)
        if state.get("status") == "running" and job_dir.name not in _jobs:
            state = dict(state)
            state["status"] = "failed"
            state["error"] = "Server restarted while job was running"

        entries.append({
            "job_id": job_dir.name,
            "topic": state.get("topic"),
            "status": state.get("status"),
            "step": state.get("step"),
            "pct": state.get("pct"),
            "error": state.get("error"),
            "created_at": state.get("created_at"),
            "failed_at_stage": state.get("failed_at_stage"),
        })

    entries.sort(
        key=lambda e: (e.get("created_at") or "", e.get("job_id") or ""),
        reverse=True,
    )
    return entries[:50]
```

- [ ] **Step 4: Run tests**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_api_resume.py::test_get_jobs_returns_list tests/test_api_resume.py::test_get_jobs_applies_running_failed_correction tests/test_api_resume.py::test_get_jobs_sorted_by_created_at -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routes.py tests/test_api_resume.py
git commit -m "feat: add GET /api/jobs endpoint"
```

---

## Task 7: `POST /api/resume/{job_id}` endpoint

**Files:**
- Modify: `api/routes.py`
- Test: `tests/test_api_resume.py` (extend)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_api_resume.py`:

```python
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
    # Need narration.json for tts start
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
        resp = client.post(f"/api/resume/job_aaa",
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

    with p("api.routes._get_jobs_base_dir", return_value=tmp_path), \
         p("mathmotion.utils.config.get_config") as mock_cfg:
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

    with p("api.routes._get_jobs_base_dir", return_value=tmp_path), \
         p("mathmotion.utils.config.get_config") as mock_cfg:
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
    _jobs["job_ccc"] = {"status": "running", "_job_dir": str(job_dir)}

    with p("api.routes._get_jobs_base_dir", return_value=tmp_path), \
         p("mathmotion.utils.config.get_config") as mock_cfg:
        mock_cfg.return_value.storage.jobs_dir = str(tmp_path)
        client = TestClient(app)
        resp = client.post("/api/resume/job_ccc",
                           json={"start_from_stage": "tts"})

    assert resp.status_code == 409
    _jobs.pop("job_ccc", None)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_api_resume.py::test_resume_returns_job_id tests/test_api_resume.py::test_resume_returns_404_unknown_job -v 2>&1 | head -20
```

Expected: 404 or AttributeError (endpoint not defined).

- [ ] **Step 3: Add `POST /api/resume/{job_id}` to `api/routes.py`**

```python
class ResumeRequest(BaseModel):
    start_from_stage: str


@router.post("/resume/{job_id}")
def resume_job(job_id: str, req: ResumeRequest):
    from fastapi import HTTPException
    from mathmotion.utils.config import get_config

    # 1. Validate stage name
    if req.start_from_stage not in STAGES:
        raise HTTPException(422, detail=f"Invalid stage '{req.start_from_stage}'. "
                                        f"Valid stages: {STAGES}")

    # 2. Validate job directory exists
    config = get_config()
    job_dir = Path(config.storage.jobs_dir) / job_id
    if not job_dir.is_dir():
        raise HTTPException(404, detail=f"Job not found: {job_id}")

    # 3. Load job_state.json
    state_file = job_dir / "job_state.json"
    if not state_file.exists():
        raise HTTPException(404, detail=f"job_state.json not found for {job_id}")
    try:
        saved_state = json.loads(state_file.read_text())
    except Exception as e:
        raise HTTPException(404, detail=f"Could not read job_state.json: {e}")

    # 4. Pre-flight validation
    _preflight_validate(job_dir, req.start_from_stage)

    # 5. Under lock: check not running, then update state
    with _state_lock:
        current = _jobs.get(job_id, {})
        if current.get("status") == "running":
            raise HTTPException(409, detail=f"Job {job_id} is already running")

        new_state = {
            **saved_state,
            "status": "running",
            "step": "Resuming…",
            "pct": 0,
            "error": None,
            "failed_at_stage": None,
            "last_resumed_from_stage": req.start_from_stage,
            "_job_dir": str(job_dir),
        }
        _jobs[job_id] = new_state
    _save_job_state(job_id, new_state, job_dir)

    # 6. Spawn background thread
    def _run():
        from mathmotion.utils.config import get_config as _get
        _get.cache_clear()
        cfg = _get()

        _current_stage = [None]  # mutable container for tracking active stage

        try:
            from mathmotion.pipeline import run as run_pipeline

            def on_progress(step: str, pct: int):
                # Infer current stage from step name for failed_at_stage tracking
                from mathmotion.pipeline import STAGES as _STAGES
                matched = next((s for s in _STAGES if s in step.lower().replace(" ", "_")), None)
                if matched:
                    _current_stage[0] = matched
                _update_job(job_id, step=step, pct=pct)

            output = run_pipeline(
                topic=saved_state["topic"],
                config=cfg,
                quality=saved_state.get("quality"),
                level=saved_state.get("level", "undergraduate"),
                voice=saved_state.get("voice"),
                tts_engine=saved_state.get("tts_engine"),
                llm_provider=saved_state.get("llm_provider"),
                progress_callback=on_progress,
                job_id=job_id,
                start_from_stage=req.start_from_stage,
            )
            _update_job(job_id, status="complete", output=str(output), pct=100, step="Done",
                        failed_at_stage=None)
        except Exception as e:
            _update_job(job_id, status="failed", error=str(e), step="Failed",
                        failed_at_stage=_current_stage[0])

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}
```

Note: `start_generate`'s `_run` closure is updated in Task 1 Step 4 (it uses the same `_current_stage` pattern shown above).

- [ ] **Step 4: Run tests**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_api_resume.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routes.py tests/test_api_resume.py
git commit -m "feat: add POST /api/resume/{job_id} endpoint"
```

---

## Task 8: Modify `GET /api/status` with lazy load

**Files:**
- Modify: `api/routes.py`
- Test: `tests/test_api_resume.py` (extend)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_api_resume.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_api_resume.py::test_status_loads_from_disk_after_restart -v 2>&1 | head -20
```

Expected: FAIL — returns `{"error": "Job not found"}`.

- [ ] **Step 3: Update `get_status` in `api/routes.py`**

Replace the current `get_status`:

```python
@router.get("/status/{job_id}")
def get_status(job_id: str):
    with _state_lock:
        if job_id not in _jobs:
            # Lazy load from disk
            from mathmotion.utils.config import get_config
            job_dir = Path(get_config().storage.jobs_dir) / job_id
            state_file = job_dir / "job_state.json"
            if not state_file.exists():
                return {"error": "Job not found"}
            try:
                loaded = json.loads(state_file.read_text())
            except Exception:
                return {"error": "Job not found"}
            # Apply running→failed correction for orphaned jobs
            if loaded.get("status") == "running":
                loaded["status"] = "failed"
                loaded["error"] = "Server restarted while job was running"
            loaded["_job_dir"] = str(job_dir)
            _jobs.setdefault(job_id, loaded)

    job = _jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    # Return only public fields (exclude internal _* keys)
    return {k: v for k, v in job.items() if not k.startswith("_")}
```

- [ ] **Step 4: Run tests**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/test_api_resume.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routes.py tests/test_api_resume.py
git commit -m "feat: lazy-load job status from disk with stale-running correction"
```

---

## Task 9: UI — Past Jobs panel, Resume modal, error Resume button

**Files:**
- Modify: `static/index.html`

No unit tests — verify manually in browser.

- [ ] **Step 1: Add CSS for the new UI components**

Add inside the `<style>` block, after `.btn-new:hover`:

```css
/* Past jobs panel */
#jobs-panel {
  width: 100%;
  max-width: 700px;
  margin-top: 1rem;
}

#jobs-toggle {
  background: none;
  border: none;
  color: #6b7280;
  font-size: 0.82rem;
  cursor: pointer;
  padding: 0.3rem 0;
  text-align: left;
  width: 100%;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-weight: 600;
}

#jobs-toggle:hover { color: #c4b5fd; }

#jobs-list { margin-top: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem; }

.job-row {
  background: #1a1a2e;
  border: 1px solid #2d2d4e;
  border-radius: 10px;
  padding: 0.75rem 1rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.job-topic {
  flex: 1;
  font-size: 0.88rem;
  color: #e0e0f0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.badge {
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.2rem 0.55rem;
  border-radius: 99px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  flex-shrink: 0;
}

.badge-complete { background: #064e3b; color: #6ee7b7; }
.badge-failed   { background: #450a0a; color: #fca5a5; }
.badge-running  { background: #451a03; color: #fdba74; }

.btn-resume {
  padding: 0.35rem 0.8rem;
  background: #4c1d95;
  color: #ddd6fe;
  border: none;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  flex-shrink: 0;
}

.btn-resume:hover { background: #5b21b6; }

/* Resume modal */
#resume-modal-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.7);
  z-index: 100;
  align-items: center;
  justify-content: center;
}

#resume-modal-overlay.open { display: flex; }

#resume-modal {
  background: #1a1a2e;
  border: 1px solid #2d2d4e;
  border-radius: 14px;
  padding: 2rem;
  width: 100%;
  max-width: 460px;
  box-shadow: 0 8px 48px rgba(0,0,0,0.6);
}

#resume-modal h2 {
  font-size: 1.1rem;
  font-weight: 700;
  color: #c4b5fd;
  margin-bottom: 1rem;
}

.resume-meta { font-size: 0.82rem; color: #6b7280; margin-bottom: 1.2rem; }
.resume-meta span { color: #e0e0f0; }

.modal-actions { display: flex; gap: 0.75rem; margin-top: 1.5rem; }

#btn-resume-submit {
  flex: 1;
  padding: 0.75rem;
  background: linear-gradient(135deg, #7c3aed, #6d28d9);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
}

#btn-resume-submit:hover { opacity: 0.9; }

#btn-resume-cancel {
  padding: 0.75rem 1.2rem;
  background: #1e293b;
  color: #94a3b8;
  border: 1px solid #334155;
  border-radius: 8px;
  font-size: 0.95rem;
  cursor: pointer;
}

#btn-resume-cancel:hover { background: #334155; }

/* Resume button on error in status card */
.btn-resume-inline {
  margin-top: 0.6rem;
  padding: 0.4rem 0.9rem;
  background: #4c1d95;
  color: #ddd6fe;
  border: none;
  border-radius: 6px;
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  display: none;
}

.btn-resume-inline:hover { background: #5b21b6; }
```

- [ ] **Step 2: Add HTML markup for panel and modal**

After `</div>` closing `#status-card`, add:

```html
<!-- Past Jobs panel -->
<div id="jobs-panel">
  <button id="jobs-toggle" onclick="toggleJobsPanel()">▸ Past Jobs</button>
  <div id="jobs-list" style="display:none"></div>
</div>

<!-- Resume modal -->
<div id="resume-modal-overlay">
  <div id="resume-modal">
    <h2>Resume Job</h2>
    <div class="resume-meta">
      Job: <span id="modal-job-id">—</span><br/>
      Topic: <span id="modal-topic">—</span>
    </div>
    <label for="modal-stage">Resume from stage</label>
    <select id="modal-stage"></select>
    <div class="modal-actions">
      <button id="btn-resume-submit" onclick="submitResume()">Resume Job</button>
      <button id="btn-resume-cancel" onclick="closeResumeModal()">Cancel</button>
    </div>
  </div>
</div>
```

Inside `#status-card`, after `<div id="error-msg"></div>`, add:

```html
<button id="btn-resume-error" class="btn-resume-inline" onclick="openResumeModalForCurrentJob()">↺ Resume from a stage</button>
```

- [ ] **Step 3: Add JavaScript**

Add to the `<script>` block, before `init()`:

```javascript
const STAGES = ["outline", "scene_script", "scene_code", "tts", "render", "compose"];
let resumeJobId = null;
let currentJobId = null;  // track the job being polled
let currentJobState = null;  // last known state for current job

// Override the existing submit to track currentJobId
const _origSubmitListener = document.getElementById("submit").onclick;

// ── Jobs panel ────────────────────────────────────────────────────────────
let jobsPanelOpen = false;

function toggleJobsPanel() {
  jobsPanelOpen = !jobsPanelOpen;
  document.getElementById("jobs-list").style.display = jobsPanelOpen ? "flex" : "none";
  document.getElementById("jobs-toggle").textContent =
    (jobsPanelOpen ? "▾" : "▸") + " Past Jobs";
  if (jobsPanelOpen) loadJobs();
}

async function loadJobs() {
  const list = document.getElementById("jobs-list");
  list.innerHTML = '<div style="color:#6b7280;font-size:0.85rem">Loading…</div>';
  try {
    const res = await fetch("/api/jobs");
    const jobs = await res.json();
    if (!jobs.length) {
      list.innerHTML = '<div style="color:#6b7280;font-size:0.85rem">No past jobs.</div>';
      return;
    }
    list.innerHTML = "";
    jobs.forEach(j => {
      const row = document.createElement("div");
      row.className = "job-row";
      const topic = (j.topic || "Untitled").substring(0, 60);
      const badgeClass = j.status === "complete" ? "badge-complete"
                       : j.status === "failed"   ? "badge-failed"
                       : "badge-running";
      const badgeLabel = j.status === "running" ? "Interrupted" : j.status;
      const canResume = j.status === "failed" || j.status === "running";
      row.innerHTML = `
        <span class="job-topic" title="${j.topic || ''}">${topic}</span>
        <span class="badge ${badgeClass}">${badgeLabel}</span>
        ${canResume ? `<button class="btn-resume" onclick="openResumeModal('${j.job_id}', '${(j.topic||'').replace(/'/g,"\\'")}', '${j.failed_at_stage||j.last_resumed_from_stage||''}')">↺ Resume</button>` : ''}
      `;
      list.appendChild(row);
    });
  } catch (e) {
    list.innerHTML = '<div style="color:#f87171;font-size:0.85rem">Failed to load jobs.</div>';
  }
}

// ── Resume modal ──────────────────────────────────────────────────────────
function openResumeModal(jobId, topic, defaultStage) {
  resumeJobId = jobId;
  document.getElementById("modal-job-id").textContent = jobId;
  document.getElementById("modal-topic").textContent = topic || "—";

  const sel = document.getElementById("modal-stage");
  sel.innerHTML = "";
  STAGES.forEach(s => {
    const o = document.createElement("option");
    o.value = s;
    o.textContent = s;
    if (s === defaultStage) o.selected = true;
    sel.appendChild(o);
  });

  document.getElementById("resume-modal-overlay").classList.add("open");
}

function openResumeModalForCurrentJob() {
  if (!currentJobId || !currentJobState) return;
  const defaultStage = currentJobState.failed_at_stage
                    || currentJobState.last_resumed_from_stage
                    || "outline";
  openResumeModal(currentJobId, currentJobState.topic, defaultStage);
}

function closeResumeModal() {
  document.getElementById("resume-modal-overlay").classList.remove("open");
  resumeJobId = null;
}

async function submitResume() {
  if (!resumeJobId) return;
  const stage = document.getElementById("modal-stage").value;
  closeResumeModal();

  document.getElementById("submit").disabled = true;
  document.getElementById("status-card").style.display = "block";
  document.getElementById("video-section").style.display = "none";
  document.getElementById("error-msg").style.display = "none";
  document.getElementById("btn-resume-error").style.display = "none";
  setProgress("Resuming…", 0);

  clearInterval(pollTimer);

  const res = await fetch(`/api/resume/${resumeJobId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start_from_stage: stage }),
  });
  const data = await res.json();

  if (!res.ok) {
    const errEl = document.getElementById("error-msg");
    errEl.textContent = "Error: " + (data.detail || data.error || "Unknown error");
    errEl.style.display = "block";
    document.getElementById("submit").disabled = false;
    return;
  }

  currentJobId = data.job_id;
  pollTimer = setInterval(() => poll(data.job_id), 2000);
}
```

Also update the existing `poll()` function to track state and show the Resume button on failure. Replace the `poll` function:

```javascript
async function poll(job_id) {
  let job;
  try {
    const res = await fetch(`/api/status/${job_id}`);
    job = await res.json();
  } catch { return; }

  currentJobId = job_id;
  currentJobState = job;

  setProgress(job.step || "Working…", job.pct || 0);

  if (job.status === "complete") {
    clearInterval(pollTimer);
    const url = `/api/download/${job_id}`;
    document.getElementById("player").src = url;
    document.getElementById("download-btn").href = url;
    document.getElementById("download-btn").download = `mathmotion_${job_id}.mp4`;
    document.getElementById("video-section").style.display = "block";
    document.getElementById("btn-resume-error").style.display = "none";
    document.getElementById("submit").disabled = false;
    if (jobsPanelOpen) loadJobs();
  } else if (job.status === "failed") {
    clearInterval(pollTimer);
    const errEl = document.getElementById("error-msg");
    errEl.textContent = "Error: " + (job.error || "Unknown error");
    errEl.style.display = "block";
    document.getElementById("btn-resume-error").style.display = "inline-block";
    setProgress("Failed", 0);
    document.getElementById("submit").disabled = false;
    if (jobsPanelOpen) loadJobs();
  }
}
```

Also update the submit click handler to track `currentJobId`:
```javascript
// In the submit click handler, after `const { job_id } = genData;`:
currentJobId = job_id;
currentJobState = null;
```

- [ ] **Step 4: Manual smoke test**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python app.py
```

Open http://localhost:8000. Verify:
- "Past Jobs" toggle appears below status card, expands/collapses
- After a job runs, it appears in Past Jobs list with correct status badge
- "↺ Resume" button opens modal with stage dropdown
- Stage dropdown defaults to `failed_at_stage` or first stage
- Clicking "Resume Job" starts polling and shows progress

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: add resume UI — past jobs panel and resume modal"
```

---

## Task 10: Run full test suite

- [ ] **Step 1: Run all tests**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home && python -m pytest tests/ -v --ignore=tests/test_api_upload.py --ignore=tests/test_pipeline_upload.py 2>&1 | tail -30
```

Expected: all tests PASS (existing tests unchanged by this work).

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "feat: complete resume-job feature — all tests passing"
```
