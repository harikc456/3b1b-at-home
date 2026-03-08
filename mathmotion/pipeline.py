import json
import logging
import uuid
from pathlib import Path
from typing import Callable, Optional

from mathmotion.llm.factory import get_provider
from mathmotion.schemas.script import GeneratedScript
from mathmotion.stages import generate, render, tts, compose
from mathmotion.stages.render import inject_actual_durations
from mathmotion.tts.factory import get_engine
from mathmotion.utils.config import Config

logger = logging.getLogger(__name__)


def run(
    topic: str,
    config: Config,
    quality: Optional[str] = None,
    level: str = "undergraduate",
    voice: Optional[str] = None,
    tts_engine: Optional[str] = None,
    llm_provider: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Path:
    def progress(step: str, pct: int) -> None:
        logger.info(f"[{pct}%] {step}")
        if progress_callback:
            progress_callback(step, pct)

    if quality:
        config.manim.default_quality = quality
    if llm_provider:
        config.llm.provider = llm_provider
    if tts_engine:
        config.tts.engine = tts_engine
    if voice:
        if config.tts.engine == "kokoro":
            config.tts.kokoro.voice = voice
        else:
            config.tts.vibevoice.voice = voice

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = Path(config.storage.jobs_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Job {job_id} | topic={topic!r} | provider={config.llm.provider} | "
                f"tts={config.tts.engine} | quality={config.manim.default_quality}")

    progress("Generating script", 10)
    provider = get_provider(config)
    script = generate.run(topic, job_dir, config, provider, level=level)

    progress("Synthesising audio", 30)
    engine = get_engine(config)
    tts.run(script, job_dir, config, engine)

    progress("Injecting durations into scene code", 55)
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

    progress("Rendering animation", 65)
    render.run(script, job_dir, config)

    progress("Composing final video", 85)
    final = compose.run(job_dir, config)

    progress("Done", 100)
    logger.info(f"Output: {final}")
    return final
