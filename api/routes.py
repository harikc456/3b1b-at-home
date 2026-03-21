import json
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()

# In-memory job tracker — fine for personal single-user use
_jobs: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    topic: str
    quality: str = "standard"
    level: str = "undergraduate"
    voice: Optional[str] = None
    tts_engine: Optional[str] = None
    llm_provider: Optional[str] = None


@router.post("/generate")
def start_generate(req: GenerateRequest):
    from mathmotion.utils.config import get_config
    config = get_config()

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _jobs[job_id] = {
        "status": "running",
        "step": "Starting…",
        "pct": 0,
        "output": None,
        "error": None,
    }

    def _run():
        # get_config uses lru_cache — clear so overrides take effect on a fresh config copy
        from mathmotion.utils.config import get_config as _get
        _get.cache_clear()
        cfg = _get()

        try:
            from mathmotion.pipeline import run as run_pipeline

            def on_progress(step: str, pct: int):
                _jobs[job_id]["step"] = step
                _jobs[job_id]["pct"] = pct

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
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["output"] = str(output)
            _jobs[job_id]["pct"] = 100
            _jobs[job_id]["step"] = "Done"
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["step"] = "Failed"

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return job


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
