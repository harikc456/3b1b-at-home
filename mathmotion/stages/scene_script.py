import json
import logging
from pathlib import Path

from mathmotion.llm.base import LLMProvider
from mathmotion.schemas.script import AllSceneScripts, TopicOutline
from mathmotion.utils.errors import LLMError

logger = logging.getLogger(__name__)


def run(
    outline: TopicOutline,
    job_dir: Path,
    config,
    provider: LLMProvider,
) -> AllSceneScripts:
    schema_json = json.dumps(AllSceneScripts.model_json_schema(), indent=2)
    system_prompt = Path("prompts/scene_script.txt").read_text().format(
        outline_json=outline.model_dump_json(indent=2),
        schema_json=schema_json,
    )

    base_prompt = "Write scene scripts for all scenes in the outline."
    user_prompt = base_prompt
    last_error = None

    for attempt in range(config.llm.max_retries + 1):
        if attempt > 0:
            logger.warning(f"Scene script retry {attempt}/{config.llm.max_retries}: {last_error}")
            user_prompt = f"{base_prompt}\n\nPrevious attempt failed — fix this: {last_error}"

        try:
            resp = provider.complete(
                system_prompt, user_prompt,
                config.llm.max_tokens, config.llm.temperature,
                response_schema=AllSceneScripts.model_json_schema(),
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
            scripts = AllSceneScripts.model_validate(data)
        except Exception as e:
            last_error = f"Schema error: {e}"
            continue

        empty = [s.id for s in scripts.scenes if not s.narration.strip()]
        if empty:
            last_error = f"Scenes with empty narration: {empty}"
            continue

        (job_dir / "scene_scripts.json").write_text(scripts.model_dump_json(indent=2))
        logger.info(f"Scene scripts: {len(scripts.scenes)} scene(s)")
        return scripts

    raise LLMError(
        f"Scene script generation failed after {config.llm.max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
