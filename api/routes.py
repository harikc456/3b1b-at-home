import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job tracker — fine for personal single-user use
_jobs: dict[str, dict] = {}

# asyncio.Lock — acquired only from async route handlers (event loop context).
# _update_job is sync and called from asyncio.to_thread; it relies on
# Python's GIL for dict safety (one pipeline per job_id at a time).
_state_lock = asyncio.Lock()


def _save_job_state(job_id: str, state: dict, job_dir: Path) -> None:
    """Atomically write job_state.json. Swallows write errors."""
    persisted = {k: v for k, v in state.items() if not k.startswith("_")}
    tmp = job_dir / "job_state.json.tmp"
    try:
        tmp.write_text(json.dumps(persisted, indent=2))
        os.replace(tmp, job_dir / "job_state.json")
    except Exception as e:
        logger.warning(f"Failed to save job_state.json for {job_id}: {e}")


def _update_job(job_id: str, **kwargs) -> None:
    """Sync state update + disk persist. Safe to call from asyncio.to_thread context."""
    if job_id not in _jobs:
        return
    _jobs[job_id].update(kwargs)
    state_copy = dict(_jobs[job_id])
    job_dir = Path(_jobs[job_id]["_job_dir"])
    _save_job_state(job_id, state_copy, job_dir)


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


def _preflight_validate(job_dir: Path, start_from_stage: Optional[str]) -> None:
    """Validate that all files required by skipped stages exist.

    Raises HTTPException(422) with a descriptive message if any are missing.
    start_from_stage=None or 'outline' means full run — no validation needed.
    """
    from fastapi import HTTPException

    from mathmotion.pipeline import STAGES  # lazy import

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
            # narration.json must have actual_duration on all segments,
            # UNLESS we're starting from compose (compose doesn't use durations)
            if start_from_stage != "compose":
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


class GenerateRequest(BaseModel):
    topic: str
    quality: str = "standard"
    level: str = "undergraduate"
    voice: Optional[str] = None
    tts_engine: Optional[str] = None
    llm_provider: Optional[str] = None


class ResumeRequest(BaseModel):
    start_from_stage: str


@router.post("/generate")
async def start_generate(req: GenerateRequest):
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
    async with _state_lock:
        _jobs[job_id] = initial_state
    _save_job_state(job_id, initial_state, job_dir)

    def _run():
        from mathmotion.utils.config import get_config as _get
        _get.cache_clear()
        cfg = _get()

        _current_stage = [None]

        try:
            from mathmotion.pipeline import run as run_pipeline

            _STAGE_KEYWORDS = {
                "outline": "generating_outline",
                "scene_script": "writing_scene_scripts",
                "scene_code": "generating_scene_code",
                "tts": "synthesising_audio",
                "render": "rendering_animation",
                "compose": "composing_final_video",
            }

            def on_progress(step: str, pct: int):
                normalized = step.lower().replace(" ", "_")
                matched = next((s for s, kw in _STAGE_KEYWORDS.items() if kw in normalized), None)
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

    asyncio.create_task(asyncio.to_thread(_run))
    return {"job_id": job_id}


@router.post("/resume/{job_id}")
async def resume_job(job_id: str, req: ResumeRequest):
    from mathmotion.pipeline import STAGES
    from mathmotion.utils.config import get_config

    # 1. Validate stage name
    if req.start_from_stage not in STAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid stage '{req.start_from_stage}'. Valid stages: {STAGES}"
        )

    # Validate job_id format to prevent path traversal
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', job_id):
        raise HTTPException(status_code=422, detail=f"Invalid job_id format: {job_id}")

    # 2. Validate job directory exists
    config = get_config()
    job_dir = Path(config.storage.jobs_dir) / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # 3. Load job_state.json
    state_file = job_dir / "job_state.json"
    if not state_file.exists():
        raise HTTPException(status_code=404, detail=f"job_state.json not found for {job_id}")
    try:
        saved_state = json.loads(state_file.read_text())
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not read job_state.json: {e}")

    # 4. Pre-flight validation — BEFORE the lock
    _preflight_validate(job_dir, req.start_from_stage)

    # 5. Under ONE lock: check not running, then update state
    async with _state_lock:
        current = _jobs.get(job_id, {})
        if current.get("status") == "running":
            raise HTTPException(status_code=409, detail=f"Job {job_id} is already running")

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

    # 7. Run pipeline as background asyncio task (no threading.Thread)
    def _run():
        from mathmotion.utils.config import get_config as _get
        _get.cache_clear()
        cfg = _get()

        _current_stage = [None]  # mutable container for tracking active stage

        try:
            from mathmotion.pipeline import run as run_pipeline

            _STAGE_KEYWORDS = {
                "outline": "generating_outline",
                "scene_script": "writing_scene_scripts",
                "scene_code": "generating_scene_code",
                "tts": "synthesising_audio",
                "render": "rendering_animation",
                "compose": "composing_final_video",
            }

            def on_progress(step: str, pct: int):
                # Infer current stage from step name for failed_at_stage tracking
                normalized = step.lower().replace(" ", "_")
                matched = next((s for s, kw in _STAGE_KEYWORDS.items() if kw in normalized), None)
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

    asyncio.create_task(asyncio.to_thread(_run))
    return {"job_id": job_id}


@router.post("/generate-from-script")
def start_generate_from_script(
    file: UploadFile = File(...),
    quality: Optional[str] = Form(None),
    tts_engine: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    llm_provider: Optional[str] = Form(None),
    level: Optional[str] = Form(None),
):
    from mathmotion.schemas.script import GeneratedScript
    from mathmotion.utils.validation import validate_script
    from mathmotion.utils.errors import ValidationError

    content = file.file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        script = validate_script(data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _jobs[job_id] = {
        "status": "running",
        "step": "Starting…",
        "pct": 0,
        "output": None,
        "error": None,
    }

    def _run():
        from mathmotion.utils.config import get_config as _get
        _get.cache_clear()
        cfg = _get()

        try:
            from mathmotion.pipeline import run as run_pipeline

            def on_progress(step: str, pct: int):
                _jobs[job_id]["step"] = step
                _jobs[job_id]["pct"] = pct

            output = run_pipeline(
                topic=script.topic,
                config=cfg,
                quality=quality,
                level=level or "undergraduate",
                tts_engine=tts_engine,
                voice=voice,
                llm_provider=llm_provider,
                progress_callback=on_progress,
                script=script,
                job_id=job_id,
            )
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["output"] = str(output)
            _jobs[job_id]["pct"] = 100
            _jobs[job_id]["step"] = "Done"
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["step"] = "Failed"

    import threading
    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    async with _state_lock:
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


@router.get("/download/{job_id}")
def download_video(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "complete":
        return {"error": "Video not ready"}
    path = Path(job["output"])
    return FileResponse(
        path, media_type="video/mp4",
        filename=f"mathmotion_{job_id}.mp4",
    )


def _get_jobs_base_dir() -> Path:
    from mathmotion.utils.config import get_config
    return Path(get_config().storage.jobs_dir)


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
            "last_resumed_from_stage": state.get("last_resumed_from_stage"),
        })

    entries.sort(
        key=lambda e: (e.get("created_at") or "", e.get("job_id") or ""),
        reverse=True,
    )
    return entries[:50]


@router.get("/config/options")
def get_options():
    from mathmotion.utils.config import get_config
    config = get_config()
    return {
        "llm_providers": list(config.llm.models),
        "tts_engines": ["kokoro", "vibevoice"],
        "kokoro_voices": list(config.tts.kokoro.available_voices),
        "vibevoice_voices": list(config.tts.vibevoice.available_voices),
        "qualities": ["draft", "standard", "high"],
        "levels": ["high_school", "undergraduate", "graduate"],
        "defaults": {
            "llm_provider": config.llm.model,
            "tts_engine": config.tts.engine,
            "quality": config.manim.default_quality,
        },
    }
