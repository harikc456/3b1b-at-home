import json
import logging
import uuid
from pathlib import Path
from typing import Callable, Optional

from mathmotion.llm.factory import get_provider
from mathmotion.schemas.script import GeneratedScript, TopicOutline
from mathmotion.stages import outline as outline_stage
from mathmotion.stages import scene_script as scene_script_stage
from mathmotion.stages import scene_code as scene_code_stage
from mathmotion.stages import render, repair, tts, compose
from mathmotion.tts.factory import get_engine
from mathmotion.utils.config import Config

logger = logging.getLogger(__name__)

STAGES = ["outline", "scene_script", "scene_code", "tts", "render", "compose"]


def _should_run(stage: str, start_from_stage: Optional[str]) -> bool:
    """Return True if this stage should execute (not be skipped)."""
    if start_from_stage is None or start_from_stage == "outline":
        return True
    return STAGES.index(stage) >= STAGES.index(start_from_stage)


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
    job_id: Optional[str] = None,
    start_from_stage: Optional[str] = None,
) -> Path:
    """Run the full mathmotion pipeline for the given topic.

    ``start_from_stage`` can be set to one of the STAGES values to skip earlier
    stages and load their outputs from disk instead.
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

    if job_id is None:
        job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = Path(config.storage.jobs_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

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
    if _should_run("tts", start_from_stage):
        # Create forensic backup before first TTS run (never overwrite).
        # Must be inside this block so we only back up the pre-TTS form,
        # not a post-TTS narration.json when TTS is skipped.
        bak_path = job_dir / "narration.json.bak"
        if not bak_path.exists():
            shutil.copy2(job_dir / "narration.json", bak_path)
        progress("Synthesising audio", 45)
        engine = get_engine(config)
        tts.run(script, job_dir, config, engine)

    # ── Inject durations ──────────────────────────────────────────────────────
    # TODO (Task 7): Remove this dead code block and all references to inject_actual_durations
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

    # ── Recalculate cue_offsets from actual Manim animation timings ───────────
    # TODO (Task 7): Remove this dead code block and all references to compute_cue_offsets
    # The LLM-supplied cue_offset values are estimates and may not match the
    # actual elapsed time in the rendered video when each narration wait fires.
    # Re-derive them by parsing the (now duration-injected) scene code so the
    # audio track built in the compose stage stays in sync with the video.
    cue_offsets_changed = False
    for scene in script.scenes:
        scene_file = scenes_dir / f"{scene.id}.py"
        if not scene_file.exists():
            continue
        seg_ids = {seg.id for seg in scene.narration_segments}
        computed = compute_cue_offsets(scene_file.read_text(), seg_ids)
        for seg in scene.narration_segments:
            if seg.id in computed and abs(computed[seg.id] - seg.cue_offset) > 0.05:
                seg.cue_offset = round(computed[seg.id], 3)
                cue_offsets_changed = True
    if cue_offsets_changed:
        (job_dir / "narration.json").write_text(script.model_dump_json(indent=2))
        logger.info("Recalculated cue_offsets saved to narration.json")

    # ── Render ────────────────────────────────────────────────────────────────
    if _should_run("render", start_from_stage):
        progress("Rendering animation", 70)
        _run_render_repair_loop(script, job_dir, config, provider)
    else:
        progress("Skipping render (using existing files)", 70)

    # ── Compose ───────────────────────────────────────────────────────────────
    progress("Composing final video", 88)
    final = compose.run(job_dir, config)

    progress("Done", 100)
    logger.info(f"Output: {final}")
    return final
