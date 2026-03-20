import ast
import json
import logging
from pathlib import Path

from mathmotion.llm.base import LLMProvider
from mathmotion.schemas.script import (
    AllSceneScripts, GeneratedScript, Scene, TopicOutline,
)
from mathmotion.utils.errors import LLMError, ValidationError
from mathmotion.utils.validation import (
    check_forbidden_imports, check_forbidden_calls,
)

logger = logging.getLogger(__name__)


def _validate_scene(scene: Scene) -> None:
    """Raises ValidationError if scene code fails any check."""
    try:
        ast.parse(scene.manim_code)
    except SyntaxError as e:
        raise ValidationError(f"Syntax error: {e}")

    bad = check_forbidden_imports(scene.manim_code)
    if bad:
        raise ValidationError(f"Forbidden imports: {bad}")

    bad = check_forbidden_calls(scene.manim_code)
    if bad:
        raise ValidationError(f"Forbidden calls: {bad}")

    if scene.class_name not in scene.manim_code:
        raise ValidationError(
            f"class_name '{scene.class_name}' not found in manim_code"
        )

    for seg in scene.narration_segments:
        if not seg.text.strip():
            raise ValidationError(f"Narration segment {seg.id} has empty text")


def _generate_scene(
    scene_script,
    outline: TopicOutline,
    config,
    provider: LLMProvider,
) -> Scene:
    """Generate Manim code for one scene. Returns Scene or raises LLMError."""
    schema_json = json.dumps(Scene.model_json_schema(), indent=2)
    system_prompt = (
        Path("prompts/scene_code.txt")
        .read_text()
        .replace("{outline_json}", outline.model_dump_json(indent=2))
        .replace("{scene_script_json}", scene_script.model_dump_json(indent=2))
        .replace("{schema_json}", schema_json)
    )

    base_prompt = f"Implement Manim code for scene: {scene_script.id}"
    user_prompt = base_prompt
    last_error = None

    for attempt in range(config.llm.max_retries + 1):
        if attempt > 0:
            logger.warning(
                f"Scene code retry {attempt}/{config.llm.max_retries} "
                f"for {scene_script.id}: {last_error}"
            )
            user_prompt = f"{base_prompt}\n\nPrevious attempt failed — fix this: {last_error}"

        try:
            resp = provider.complete(
                system_prompt, user_prompt,
                config.llm.max_tokens, config.llm.temperature,
                response_schema=Scene.model_json_schema(),
            )
        except LLMError as e:
            last_error = str(e)
            continue

        try:
            data = json.loads(resp.content)
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
            continue

        try:
            scene = Scene.model_validate(data)
        except Exception as e:
            last_error = f"Schema error: {e}"
            continue

        try:
            _validate_scene(scene)
        except ValidationError as e:
            last_error = str(e)
            continue

        return scene

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
    """Generate Manim code for all scenes. Raises LLMError if any scene fails."""
    generated: list[Scene] = []
    failures: list[str] = []

    for scene_script in scripts.scenes:
        try:
            scene = _generate_scene(scene_script, outline, config, provider)
            generated.append(scene)
            logger.info(f"Generated code for scene: {scene_script.id}")
        except LLMError as e:
            logger.error(f"Failed to generate code for {scene_script.id}: {e}")
            failures.append(scene_script.id)

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
    (job_dir / "narration.json").write_text(script.model_dump_json(indent=2))

    logger.info(f"Generated {len(generated)} scene(s) — files written to {scenes_dir.resolve()}")
    return script
