import json
import logging
from pathlib import Path

from mathmotion.llm.base import LLMProvider
from mathmotion.schemas.script import AllSceneScripts, SceneOutlineItem, SceneScript, TopicOutline
from mathmotion.utils.errors import LLMError

logger = logging.getLogger(__name__)


def _generate_scene_script(
    scene_item: SceneOutlineItem,
    outline: TopicOutline,
    config,
    provider: LLMProvider,
    prompt_template: str,
) -> SceneScript:
    """Generate a script for one scene. Returns SceneScript or raises LLMError."""
    schema_json = json.dumps(SceneScript.model_json_schema(), indent=2)
    system_prompt = (
        prompt_template
        .replace("{outline_json}", outline.model_dump_json(indent=2))
        .replace("{scene_item_json}", scene_item.model_dump_json(indent=2))
        .replace("{schema_json}", schema_json)
    )

    base_prompt = f"Write the script for scene: {scene_item.id} — {scene_item.title}"
    user_prompt = base_prompt
    last_error = None

    for attempt in range(config.llm.max_retries + 1):
        if attempt > 0:
            logger.warning(
                f"Scene script retry {attempt}/{config.llm.max_retries} "
                f"for {scene_item.id}: {last_error}"
            )
            user_prompt = f"{base_prompt}\n\nPrevious attempt failed — fix this: {last_error}"

        try:
            resp = provider.complete(
                system_prompt, user_prompt,
                config.llm.max_tokens, config.llm.temperature,
                response_schema=SceneScript.model_json_schema(),
            )
        except LLMError as e:
            last_error = str(e)
            continue

        try:
            data = json.loads(resp.content)
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
            continue

        data["id"] = scene_item.id  # enforce ID — don't trust LLM
        try:
            script = SceneScript.model_validate(data)
        except Exception as e:
            last_error = f"Schema error: {e}"
            continue

        if not script.narration.strip():
            last_error = f"Empty narration for scene {scene_item.id}"
            continue

        return script

    raise LLMError(
        f"Scene script generation failed for {scene_item.id} after "
        f"{config.llm.max_retries + 1} attempts. Last error: {last_error}"
    )


def run(
    outline: TopicOutline,
    job_dir: Path,
    config,
    provider: LLMProvider,
) -> AllSceneScripts:
    prompt_template = Path("prompts/scene_script.txt").read_text()
    generated: list[SceneScript] = []
    failures: list[str] = []

    for scene_item in outline.scenes:
        try:
            script = _generate_scene_script(scene_item, outline, config, provider, prompt_template)
            generated.append(script)
            logger.info(f"Generated script for scene: {scene_item.id}")
        except LLMError as e:
            logger.error(f"Failed to generate script for {scene_item.id}: {e}")
            failures.append(scene_item.id)

    if failures:
        raise LLMError(
            f"Scene script generation failed for {len(failures)} scene(s): {failures}"
        )

    scripts = AllSceneScripts(
        title=outline.title,
        topic=outline.topic,
        scenes=generated,
    )
    (job_dir / "scene_scripts.json").write_text(scripts.model_dump_json(indent=2))
    logger.info(f"Scene scripts: {len(generated)} scene(s)")
    return scripts
