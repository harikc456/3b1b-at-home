import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mathmotion.llm.base import LLMProvider
from mathmotion.schemas.script import (
    AllSceneScripts, GeneratedScript, NarrationSegment, Scene, TopicOutline,
)
from mathmotion.utils.errors import LLMError, ValidationError
from mathmotion.utils.validation import validate_scene_item

logger = logging.getLogger(__name__)


def _strip_fences(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


def _parse_code_to_scene(scene_id: str, code: str) -> Scene:
    """Parse raw Python code into a Scene, extracting class name and voiceover segments."""
    code = _strip_fences(code)

    m = re.search(r"class (Scene_\w+)", code)
    if not m:
        raise ValueError("No Scene_ class found in generated code")
    class_name = m.group(1)

    pattern = r'self\.voiceover\(\s*(?:text\s*=\s*)?(["\'])(.*?)\1\s*\)'
    texts = [m.group(2) for m in re.finditer(pattern, code, re.DOTALL)]

    segments = [
        NarrationSegment(id=f"seg_{i}", text=text)
        for i, text in enumerate(texts)
    ]

    return Scene(id=scene_id, class_name=class_name, manim_code=code, narration_segments=segments)


def _generate_scene(
    scene_script,
    outline: TopicOutline,
    config,
    provider: LLMProvider,
    prompt_template: str,
) -> tuple[Scene, list[dict]]:
    """Generate Manim code for one scene. Returns (Scene, error_records) or raises LLMError."""
    outline_context = {
        "title": outline.title,
        "topic": outline.topic,
        "scenes": [{"id": s.id, "title": s.title} for s in outline.scenes],
    }

    system_prompt = (
        prompt_template
        .replace("{outline_json}", json.dumps(outline_context, indent=2))
        .replace("{scene_script_json}", scene_script.model_dump_json(indent=2))
    )

    base_prompt = f"Implement Manim code for scene: {scene_script.id}"
    user_prompt = base_prompt
    last_error = None
    error_records: list[dict] = []

    for attempt in range(config.llm.max_retries + 1):
        if attempt > 0:
            logger.warning(
                f"Scene code retry {attempt}/{config.llm.max_retries} "
                f"for {scene_script.id}: {last_error}"
            )
            user_prompt = f"{base_prompt}\n\nPrevious attempt failed — fix this: {last_error}"

        raw_response = None
        try:
            resp = provider.complete(
                system_prompt, user_prompt,
                config.llm.max_tokens, config.llm.temperature,
                json_mode=False,
            )
            raw_response = resp.content
        except LLMError as e:
            last_error = str(e)
            error_records.append({"scene_id": scene_script.id, "attempt": attempt, "error": last_error, "raw_response": None})
            continue

        try:
            scene = _parse_code_to_scene(scene_script.id, raw_response)
        except Exception as e:
            last_error = f"Parse error: {e}"
            error_records.append({"scene_id": scene_script.id, "attempt": attempt, "error": last_error, "raw_response": raw_response})
            continue

        try:
            validate_scene_item(scene)
        except ValidationError as e:
            last_error = str(e)
            error_records.append({"scene_id": scene_script.id, "attempt": attempt, "error": last_error, "raw_response": raw_response})
            continue

        return scene, error_records

    raise LLMError(
        f"Code generation failed for {scene_script.id} after "
        f"{config.llm.max_retries + 1} attempts. Last error: {last_error}"
    )


def run(
    scripts: AllSceneScripts,
    outline: TopicOutline,
    job_dir: Path,
    config,
    provider: LLMProvider,
) -> GeneratedScript:
    """Generate Manim code for all scenes. Sequential with idempotency."""
    prompt_template = Path("prompts/scene_code.txt").read_text()

    narration_file = job_dir / "narration.json"
    existing_scenes = {}
    if narration_file.exists():
        try:
            data = json.loads(narration_file.read_text())
            existing_scenes = {s["id"]: Scene.model_validate(s) for s in data.get("scenes", [])}
            logger.info(f"Loaded {len(existing_scenes)} existing scene(s) from disk")
        except Exception:
            logger.warning("Failed to load existing narration, re-generating all")

    generated_map: dict[str, Scene] = dict(existing_scenes)
    failures: list[str] = []
    all_errors: list[dict] = []

    pending = [s for s in scripts.scenes if s.id not in existing_scenes]
    workers = min(config.llm.max_parallel_scenes, len(pending)) if pending else 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_generate_scene, s, outline, config, provider, prompt_template): s
            for s in pending
        }
        for future in as_completed(futures):
            scene_script = futures[future]
            try:
                scene, error_records = future.result()
                generated_map[scene_script.id] = scene
                all_errors.extend(error_records)
                logger.info(f"Generated code for scene: {scene_script.id}")
            except LLMError as e:
                logger.error(f"Failed to generate code for {scene_script.id}: {e}")
                failures.append(scene_script.id)

    if all_errors:
        errors_file = job_dir / "scene_code_errors.jsonl"
        with errors_file.open("a") as f:
            for record in all_errors:
                f.write(json.dumps(record) + "\n")
        logger.info(f"Wrote {len(all_errors)} error record(s) to {errors_file.name}")

    # Preserve original scene order
    generated = [generated_map[s.id] for s in scripts.scenes if s.id in generated_map]

    if failures:
        raise LLMError(
            f"Code generation failed for {len(failures)} scene(s): {failures}"
        )

    script = GeneratedScript(
        title=outline.title,
        topic=outline.topic,
        scenes=generated,
    )

    scenes_dir = job_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    for scene in script.scenes:
        (scenes_dir / f"{scene.id}.py").write_text(scene.manim_code)
    narration_file.write_text(script.model_dump_json(indent=2))

    logger.info(f"Generated {len(generated)} scene(s) — files written to {scenes_dir.resolve()}")
    return script
