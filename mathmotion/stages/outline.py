import json
import logging
from pathlib import Path

from mathmotion.llm.base import LLMProvider
from mathmotion.schemas.script import TopicOutline
from mathmotion.utils.errors import LLMError, ValidationError

logger = logging.getLogger(__name__)

DOMAIN_MAP = {
    "calculus":       ["derivative", "integral", "limit", "series", "differential"],
    "linear_algebra": ["matrix", "vector", "eigenvalue", "eigenvector", "linear", "transform"],
    "geometry":       ["hyperbolic", "curve", "manifold", "surface", "geodesic"],
    "topology":       ["homotopy", "homology", "fundamental group", "knot"],
}


def _detect_domains(topic: str) -> list[str]:
    t = topic.lower()
    return [d for d, kws in DOMAIN_MAP.items() if any(kw in t for kw in kws)]


def run(
    topic: str,
    job_dir: Path,
    config,
    provider: LLMProvider,
    level: str = "undergraduate",
) -> TopicOutline:
    domains = _detect_domains(topic)
    domain_hints = ""
    for d in domains:
        hint_file = Path(f"prompts/domain_hints/{d}.txt")
        if hint_file.exists():
            domain_hints += hint_file.read_text() + "\n"
    if not domain_hints:
        domain_hints = "General mathematics."

    schema_json = json.dumps(TopicOutline.model_json_schema(), indent=2)
    system_prompt = Path("prompts/outline.txt").read_text().format(
        topic=topic,
        level=level,
        domain_hints=domain_hints,
        schema_json=schema_json,
    )

    base_prompt = f"Plan the scene structure for a video about: {topic}"
    user_prompt = base_prompt
    last_error = None

    for attempt in range(config.llm.max_retries + 1):
        if attempt > 0:
            logger.warning(f"Outline retry {attempt}/{config.llm.max_retries}: {last_error}")
            user_prompt = f"{base_prompt}\n\nPrevious attempt failed — fix this: {last_error}"

        try:
            resp = provider.complete(
                system_prompt, user_prompt,
                config.llm.max_tokens, config.llm.temperature,
                response_schema=TopicOutline.model_json_schema(),
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
            outline = TopicOutline.model_validate(data)
        except Exception as e:
            last_error = f"Schema error: {e}"
            continue

        if not outline.scenes:
            last_error = "Outline has no scenes"
            continue

        (job_dir / "outline.json").write_text(outline.model_dump_json(indent=2))
        logger.info(f"Outline: {len(outline.scenes)} scene(s) for: {topic!r}")
        return outline

    raise LLMError(
        f"Outline generation failed after {config.llm.max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
