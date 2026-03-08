import ast
import json
import logging
from pathlib import Path

from mathmotion.llm.base import LLMProvider
from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import LLMError, ValidationError
from mathmotion.utils.validation import check_forbidden_imports, check_forbidden_calls

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


def _validate(data: dict) -> GeneratedScript:
    """Run all validation checks. Raises ValidationError with a message suitable for retry."""
    try:
        script = GeneratedScript.model_validate(data)
    except Exception as e:
        raise ValidationError(f"Schema error: {e}")

    for scene in script.scenes:
        try:
            ast.parse(scene.manim_code)
        except SyntaxError as e:
            raise ValidationError(f"Scene {scene.id} syntax error: {e}")

        bad = check_forbidden_imports(scene.manim_code)
        if bad:
            raise ValidationError(f"Scene {scene.id} has forbidden imports: {bad}")

        bad = check_forbidden_calls(scene.manim_code)
        if bad:
            raise ValidationError(f"Scene {scene.id} has forbidden calls: {bad}")

        if scene.class_name not in scene.manim_code:
            raise ValidationError(
                f"Scene {scene.id}: class_name '{scene.class_name}' not found in manim_code"
            )

        for seg in scene.narration_segments:
            if not seg.text.strip():
                raise ValidationError(f"Narration segment {seg.id} has empty text")

    return script


def run(
    topic: str,
    job_dir: Path,
    config,
    provider: LLMProvider,
    level: str = "undergraduate",
) -> GeneratedScript:
    domains = _detect_domains(topic)
    domain_hints = ""
    for d in domains:
        hint_file = Path(f"prompts/domain_hints/{d}.txt")
        if hint_file.exists():
            domain_hints += hint_file.read_text() + "\n"
    if not domain_hints:
        domain_hints = "General mathematics."

    schema_json = json.dumps(GeneratedScript.model_json_schema(), indent=2)
    system_prompt = Path("prompts/system_prompt.txt").read_text().format(
        topic=topic,
        level=level,
        domain_hints=domain_hints,
        schema_json=schema_json,
    )

    base_prompt = f"Generate a Manim animation script for: {topic}"
    user_prompt = base_prompt
    last_error = None

    for attempt in range(config.llm.max_retries + 1):
        if attempt > 0:
            logger.warning(f"Retry {attempt}/{config.llm.max_retries}: {last_error}")
            user_prompt = f"{base_prompt}\n\nPrevious attempt failed — fix this: {last_error}"

        try:
            resp = provider.complete(
                system_prompt, user_prompt,
                config.llm.max_tokens, config.llm.temperature,
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
            script = _validate(data)
        except ValidationError as e:
            last_error = str(e)
            continue

        # Write scene files
        scenes_dir = job_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        for scene in script.scenes:
            (scenes_dir / f"{scene.id}.py").write_text(scene.manim_code)

        # Write narration.json
        (job_dir / "narration.json").write_text(script.model_dump_json(indent=2))

        logger.info(f"Generated {len(script.scenes)} scene(s) for: {topic!r}")
        return script

    raise LLMError(
        f"Script generation failed after {config.llm.max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
