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


class GenerateRequest(BaseModel):
    topic: str
    quality: str = "standard"
    level: str = "undergraduate"
    voice: Optional[str] = None
    tts_engine: Optional[str] = None
    llm_provider: Optional[str] = None


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
    job = _jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
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
