import json
import logging
import uuid
from pathlib import Path
from typing import Callable, Optional

from mathmotion.llm.factory import get_provider
from mathmotion.schemas.script import GeneratedScript
from mathmotion.stages import generate, render, repair, tts, compose
from mathmotion.stages.render import inject_actual_durations
from mathmotion.tts.factory import get_engine
from mathmotion.utils.config import Config

logger = logging.getLogger(__name__)


def _run_render_repair_loop(
    script: GeneratedScript,
    job_dir: Path,
    config,
    provider,
) -> dict:
    """
    Render all scenes, repair failures with LLM, repeat up to repair_max_retries.
    Any scene that still fails after all attempts gets the _fallback title card.
    """
    from mathmotion.stages.render import _fallback, try_render_all

    quality = config.manim.default_quality
    scenes_dir = job_dir / "scenes"
    render_dir = scenes_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    scene_map = {s.id: s for s in script.scenes}
    remaining = list(script.scenes)
    results: dict[str, Path] = {}

    for attempt in range(config.llm.repair_max_retries + 1):
        if not remaining:
            break

        successes, failures = try_render_all(remaining, scenes_dir, render_dir, quality, config)
        results.update(successes)
        remaining = [scene_map[sid] for sid in failures]

        if not remaining:
            break

        if attempt < config.llm.repair_max_retries:
            logger.info(
                f"Repair attempt {attempt + 1}/{config.llm.repair_max_retries}: "
                f"{len(remaining)} scene(s) need fixing"
            )
            for scene in remaining:
                try:
                    repair.fix_scene(
                        scenes_dir / f"{scene.id}.py",
                        failures[scene.id],
                        provider,
                    )
                except Exception as e:
                    logger.warning(f"LLM repair failed for {scene.id}: {e} — will retry as-is")

    for scene in remaining:
        logger.error(f"Scene {scene.id} exhausted repair attempts — using fallback title card")
        results[scene.id] = _fallback(scene, scenes_dir, render_dir, config)

    return results


def run(
    topic: str,
    config: Config,
    quality: Optional[str] = None,
    level: str = "undergraduate",
    voice: Optional[str] = None,
    tts_engine: Optional[str] = None,
    llm_provider: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    script: Optional[GeneratedScript] = None,
) -> Path:
    """Run the full mathmotion pipeline for the given topic.

    When ``script`` is provided, the generate stage is skipped; scene files and
    ``narration.json`` are written directly from the provided object.  ``provider``
    is still constructed because it is needed for the repair loop.
    """
    def progress(step: str, pct: int) -> None:
        logger.info(f"[{pct}%] {step}")
        if progress_callback:
            progress_callback(step, pct)

    if quality:
        config.manim.default_quality = quality
    if llm_provider:
        config.llm.model = llm_provider
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

    logger.info(f"Job {job_id} | topic={topic!r} | model={config.llm.model} | "
                f"tts={config.tts.engine} | quality={config.manim.default_quality}")

    progress("Preparing script" if script is not None else "Generating script", 10)
    provider = get_provider(config)
    if script is None:
        script = generate.run(topic, job_dir, config, provider, level=level)
    else:
        # Caller is responsible for validating the script (e.g. via mathmotion.utils.validation.validate_script()).
        # We trust the provided script and write files directly.
        (job_dir / "scenes").mkdir(parents=True, exist_ok=True)
        for scene in script.scenes:
            (job_dir / "scenes" / f"{scene.id}.py").write_text(scene.manim_code)
        (job_dir / "narration.json").write_text(script.model_dump_json(indent=2))

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
    _run_render_repair_loop(script, job_dir, config, provider)

    progress("Composing final video", 85)
    final = compose.run(job_dir, config)

    progress("Done", 100)
    logger.info(f"Output: {final}")
    return final
