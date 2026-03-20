# Multi-Step Video Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-shot `generate.py` stage with three focused stages — outline, scene_script, scene_code — each persisting its output to disk before the next stage runs.

**Architecture:** Three new stage modules (`outline.py`, `scene_script.py`, `scene_code.py`) replace `generate.py`. Each stage has its own prompt file and Pydantic schema. `validate_script` moves from `generate.py` to `validation.py`. `pipeline.py` chains the three stages sequentially, preserving all downstream stages (TTS → render → compose) unchanged.

**Tech Stack:** Python 3.13, Pydantic v2, LiteLLM (`provider.complete()`), pytest + `unittest.mock`

**Spec:** `docs/superpowers/specs/2026-03-20-multi-step-generation-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `mathmotion/schemas/script.py` | Add 7 new models: `SceneOutlineItem`, `TopicOutline`, `AnimationObject`, `AnimationStep`, `AnimationDescription`, `SceneScript`, `AllSceneScripts` |
| Modify | `mathmotion/utils/validation.py` | Add `validate_script()` (moved from `generate.py`) |
| Create | `prompts/outline.txt` | Step 1 system prompt |
| Create | `prompts/scene_script.txt` | Step 2 system prompt |
| Create | `prompts/scene_code.txt` | Step 3 system prompt |
| Create | `mathmotion/stages/outline.py` | Step 1 stage: outline generation |
| Create | `mathmotion/stages/scene_script.py` | Step 2 stage: scene script generation |
| Create | `mathmotion/stages/scene_code.py` | Step 3 stage: per-scene code generation |
| Modify | `mathmotion/pipeline.py` | Wire three new stages, update progress labels |
| Modify | `api/routes.py` | Update `validate_script` import path |
| Delete | `mathmotion/stages/generate.py` | Superseded |
| Delete | `prompts/system_prompt.txt` | Superseded |
| Create | `tests/test_stage_outline.py` | Tests for `outline.py` |
| Create | `tests/test_stage_scene_script.py` | Tests for `scene_script.py` |
| Create | `tests/test_stage_scene_code.py` | Tests for `scene_code.py` |
| Modify | `tests/test_pipeline_upload.py` | Update patches for deleted `generate.run` |

---

## Task 1: Add New Schemas

**Files:**
- Modify: `mathmotion/schemas/script.py`
- Test: `tests/test_schemas_new.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_schemas_new.py`:

```python
import pytest
from pydantic import ValidationError


def test_topic_outline_valid():
    from mathmotion.schemas.script import TopicOutline, SceneOutlineItem
    outline = TopicOutline(
        title="Derivatives",
        topic="derivatives",
        level="undergraduate",
        scenes=[SceneOutlineItem(id="scene_1", title="Intro", purpose="Introduce concept", order=1)],
    )
    assert outline.level == "undergraduate"
    assert outline.scenes[0].id == "scene_1"


def test_topic_outline_rejects_empty_scenes():
    from mathmotion.schemas.script import TopicOutline
    with pytest.raises(ValidationError):
        TopicOutline(title="X", topic="x", level="undergraduate", scenes="not-a-list")


def test_all_scene_scripts_valid():
    from mathmotion.schemas.script import (
        AllSceneScripts, SceneScript, AnimationDescription,
        AnimationObject, AnimationStep,
    )
    scripts = AllSceneScripts(
        title="Derivatives",
        topic="derivatives",
        scenes=[
            SceneScript(
                id="scene_1",
                title="Intro",
                narration="Today we learn about derivatives.",
                animation_description=AnimationDescription(
                    objects=[AnimationObject(id="title", type="Text", color="WHITE", initial_position="CENTER")],
                    sequence=[AnimationStep(action="FadeIn", target="title", timing="start", parameters={})],
                    notes="",
                ),
            )
        ],
    )
    assert scripts.scenes[0].narration == "Today we learn about derivatives."


def test_animation_step_parameters_typed():
    from mathmotion.schemas.script import AnimationStep
    step = AnimationStep(
        action="MoveTo",
        target="circle_1",
        timing="after_narration_segment_1",
        parameters={"scale": 2, "color": "BLUE"},
    )
    assert step.parameters["scale"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_schemas_new.py -v
```
Expected: `ImportError` — `SceneOutlineItem`, `TopicOutline`, etc. do not exist yet.

- [ ] **Step 3: Add new models to `mathmotion/schemas/script.py`**

Append after the existing `GeneratedScript` class:

```python
# ── Step 1 schemas ────────────────────────────────────────────────────────────

class SceneOutlineItem(BaseModel):
    id: str
    title: str
    purpose: str
    order: int


class TopicOutline(BaseModel):
    title: str
    topic: str
    level: str
    scenes: list[SceneOutlineItem]


# ── Step 2 schemas ────────────────────────────────────────────────────────────

class AnimationObject(BaseModel):
    id: str
    type: str
    color: str
    initial_position: str


class AnimationStep(BaseModel):
    action: str
    target: str
    timing: str
    parameters: dict[str, str | float | int]


class AnimationDescription(BaseModel):
    objects: list[AnimationObject]
    sequence: list[AnimationStep]
    notes: str


class SceneScript(BaseModel):
    id: str
    title: str
    narration: str
    animation_description: AnimationDescription


class AllSceneScripts(BaseModel):
    title: str
    topic: str
    scenes: list[SceneScript]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_schemas_new.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add mathmotion/schemas/script.py tests/test_schemas_new.py
git commit -m "feat: add outline, scene-script, and animation schemas"
```

---

## Task 2: Move `validate_script` to `validation.py`

**Files:**
- Modify: `mathmotion/utils/validation.py`
- Test: `tests/test_validation.py` (modify — add 2 tests)

The function stays identical; only its home changes. It will be removed from `generate.py` in Task 9 when `generate.py` is deleted.

- [ ] **Step 1: Write the failing test**

Add to the end of `tests/test_validation.py`:

```python
def test_validate_script_importable_from_validation():
    from mathmotion.utils.validation import validate_script
    assert callable(validate_script)


def test_validate_script_rejects_invalid_schema():
    from mathmotion.utils.validation import validate_script
    from mathmotion.utils.errors import ValidationError
    with pytest.raises(ValidationError):
        validate_script({"title": "X"})  # missing required fields
```

Also add `import pytest` at the top of `tests/test_validation.py` if not already present.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_validation.py::test_validate_script_importable_from_validation -v
```
Expected: `ImportError` — `validate_script` not in `validation.py` yet.

- [ ] **Step 3: Add `validate_script` to `mathmotion/utils/validation.py`**

Add these two imports after the existing `import ast` line at the top of `mathmotion/utils/validation.py`:

```python
from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import ValidationError
```

Then append the function at the end of the file:

```python
def validate_script(data: dict) -> GeneratedScript:
    """Validate a GeneratedScript dict. Raises ValidationError with a message suitable for retry."""
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
```

Note: `ast` is already at the top of `validation.py` — do not add it again.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_validation.py -v
```
Expected: all tests PASSED (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add mathmotion/utils/validation.py tests/test_validation.py
git commit -m "feat: move validate_script into utils/validation.py"
```

---

## Task 3: Write Prompt Files

**Files:**
- Create: `prompts/outline.txt`
- Create: `prompts/scene_script.txt`
- Create: `prompts/scene_code.txt`

No unit tests — prompts are configuration. They are exercised by the stage tests in Tasks 4–6.

- [ ] **Step 1: Create `prompts/outline.txt`**

```
You are an expert mathematical educator planning an educational video.

TOPIC: {topic}
LEVEL: {level}

DOMAIN HINTS:
{domain_hints}

Break this topic into an ordered sequence of scenes that teach the concept progressively and completely. Each scene must have a single clear pedagogical purpose. Choose the number of scenes based on complexity — no padding, no premature truncation.

RULES:
1. Respond with a single JSON object only. No markdown fences. No explanation.
2. Conform exactly to this JSON schema:
{schema_json}
3. Scene IDs must follow the pattern: "scene_1", "scene_2", etc.
4. Order scenes so each builds on the previous.
```

- [ ] **Step 2: Create `prompts/scene_script.txt`**

```
You are an expert mathematical animator and educator writing scripts for an educational video.

OUTLINE:
{outline_json}

Write a complete script for every scene. For each scene produce:
1. Full spoken narration as natural conversational text.
2. A fully prescriptive animation description specifying exactly what appears on screen.

NARRATION RULES:
- Spoken language only — no LaTeX, no Unicode math symbols.
- Write "x squared plus one" not "x² + 1". Write "pi" not "π".
- Appropriate for the level stated in the outline.

ANIMATION DESCRIPTION RULES:
- List every object on screen: type (Manim class name), color (hex like "#FF6B6B" or named Manim color like "BLUE"), initial_position (Manim coordinate expression like "CENTER", "UP*2 + LEFT*3").
- List every animation step: action (FadeIn/FadeOut/MoveTo/Highlight/Dim/Write), target (object id), timing (e.g. "after_narration_segment_2", "simultaneously_with_step_3"), and parameters.
- Be fully prescriptive — the code generator must not need to guess anything.

OUTPUT RULES:
1. Respond with a single JSON object only. No markdown fences. No explanation.
2. Conform exactly to this JSON schema:
{schema_json}
```

- [ ] **Step 3: Create `prompts/scene_code.txt`**

```
You are an expert Manim animation programmer. Implement a complete Manim scene from a detailed script.

VIDEO OUTLINE (context — this scene's position in the full video):
{outline_json}

SCENE SCRIPT TO IMPLEMENT:
{scene_script_json}

RULES:
1. Respond with a single JSON object only. No markdown fences. No explanation.
2. Conform exactly to this JSON schema:
{schema_json}
3. manim_code must be a complete, self-contained Python file with all imports.
4. The class name must start with "Scene_" and exactly match the class_name field.
5. Split the narration into narration_segments at natural pause points.
6. Every narration-aligned self.wait() must be preceded by: # WAIT:{segment_id}
7. self.wait() values must always be 1 — replaced with real TTS durations before rendering.
8. Narration text must be spoken language — no LaTeX, no Unicode math symbols.
9. Forbidden imports: os, sys, subprocess, socket, urllib, requests, httpx.
10. Forbidden calls: open(), exec(), eval(), __import__().
11. Never use `from manim import *` — import only the specific classes you use.
12. Implement the animation exactly as described in animation_description.
```

- [ ] **Step 4: Commit**

```bash
git add prompts/outline.txt prompts/scene_script.txt prompts/scene_code.txt
git commit -m "feat: add three-step generation prompt files"
```

---

## Task 4: Implement `outline.py`

**Files:**
- Create: `mathmotion/stages/outline.py`
- Create: `tests/test_stage_outline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stage_outline.py`:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


VALID_OUTLINE_JSON = json.dumps({
    "title": "Understanding Derivatives",
    "topic": "derivatives",
    "level": "undergraduate",
    "scenes": [
        {"id": "scene_1", "title": "Introduction", "purpose": "Introduce concept", "order": 1},
        {"id": "scene_2", "title": "Formal Definition", "purpose": "Define derivative", "order": 2},
    ],
})


def _make_provider(content: str) -> MagicMock:
    provider = MagicMock()
    resp = MagicMock()
    resp.content = content
    provider.complete.return_value = resp
    return provider


def _make_config(max_retries: int = 1) -> MagicMock:
    cfg = MagicMock()
    cfg.llm.max_retries = max_retries
    cfg.llm.max_tokens = 4096
    cfg.llm.temperature = 0.2
    return cfg


def test_run_returns_topic_outline(tmp_path):
    from mathmotion.stages.outline import run
    from mathmotion.schemas.script import TopicOutline

    provider = _make_provider(VALID_OUTLINE_JSON)
    cfg = _make_config()

    result = run("derivatives", tmp_path, cfg, provider, level="undergraduate")

    assert isinstance(result, TopicOutline)
    assert result.topic == "derivatives"
    assert len(result.scenes) == 2
    assert result.scenes[0].id == "scene_1"


def test_run_persists_outline_json(tmp_path):
    from mathmotion.stages.outline import run

    provider = _make_provider(VALID_OUTLINE_JSON)
    cfg = _make_config()

    run("derivatives", tmp_path, cfg, provider)

    outline_file = tmp_path / "outline.json"
    assert outline_file.exists()
    data = json.loads(outline_file.read_text())
    assert data["topic"] == "derivatives"


def test_run_passes_schema_to_provider(tmp_path):
    from mathmotion.stages.outline import run
    from mathmotion.schemas.script import TopicOutline

    provider = _make_provider(VALID_OUTLINE_JSON)
    cfg = _make_config()

    run("derivatives", tmp_path, cfg, provider)

    call_kwargs = provider.complete.call_args.kwargs
    assert call_kwargs["response_schema"] == TopicOutline.model_json_schema()


def test_run_retries_on_invalid_json(tmp_path):
    from mathmotion.stages.outline import run

    provider = MagicMock()
    bad_resp = MagicMock()
    bad_resp.content = "not json"
    good_resp = MagicMock()
    good_resp.content = VALID_OUTLINE_JSON
    provider.complete.side_effect = [bad_resp, good_resp]

    cfg = _make_config(max_retries=1)
    result = run("derivatives", tmp_path, cfg, provider)

    assert result.topic == "derivatives"
    assert provider.complete.call_count == 2


def test_run_raises_llm_error_after_max_retries(tmp_path):
    from mathmotion.stages.outline import run
    from mathmotion.utils.errors import LLMError

    provider = _make_provider("not valid json at all")
    cfg = _make_config(max_retries=0)

    try:
        run("derivatives", tmp_path, cfg, provider)
        assert False, "Should have raised LLMError"
    except LLMError as e:
        assert "failed" in str(e).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_stage_outline.py -v
```
Expected: `ModuleNotFoundError` — `mathmotion.stages.outline` does not exist.

- [ ] **Step 3: Implement `mathmotion/stages/outline.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_stage_outline.py -v
```
Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add mathmotion/stages/outline.py tests/test_stage_outline.py
git commit -m "feat: implement outline stage (step 1 of 3)"
```

---

## Task 5: Implement `scene_script.py`

**Files:**
- Create: `mathmotion/stages/scene_script.py`
- Create: `tests/test_stage_scene_script.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stage_scene_script.py`:

```python
import json
from unittest.mock import MagicMock


VALID_SCRIPTS_JSON = json.dumps({
    "title": "Understanding Derivatives",
    "topic": "derivatives",
    "scenes": [
        {
            "id": "scene_1",
            "title": "Introduction",
            "narration": "Today we learn about derivatives.",
            "animation_description": {
                "objects": [{"id": "title", "type": "Text", "color": "WHITE", "initial_position": "CENTER"}],
                "sequence": [{"action": "FadeIn", "target": "title", "timing": "start", "parameters": {}}],
                "notes": "",
            },
        }
    ],
})


def _make_outline() -> "TopicOutline":
    from mathmotion.schemas.script import TopicOutline, SceneOutlineItem
    return TopicOutline(
        title="Understanding Derivatives",
        topic="derivatives",
        level="undergraduate",
        scenes=[SceneOutlineItem(id="scene_1", title="Introduction", purpose="Introduce", order=1)],
    )


def _make_provider(content: str) -> MagicMock:
    provider = MagicMock()
    resp = MagicMock()
    resp.content = content
    provider.complete.return_value = resp
    return provider


def _make_config(max_retries: int = 1) -> MagicMock:
    cfg = MagicMock()
    cfg.llm.max_retries = max_retries
    cfg.llm.max_tokens = 4096
    cfg.llm.temperature = 0.2
    return cfg


def test_run_returns_all_scene_scripts(tmp_path):
    from mathmotion.stages.scene_script import run
    from mathmotion.schemas.script import AllSceneScripts

    provider = _make_provider(VALID_SCRIPTS_JSON)
    cfg = _make_config()
    outline = _make_outline()

    result = run(outline, tmp_path, cfg, provider)

    assert isinstance(result, AllSceneScripts)
    assert len(result.scenes) == 1
    assert result.scenes[0].narration == "Today we learn about derivatives."


def test_run_persists_scene_scripts_json(tmp_path):
    from mathmotion.stages.scene_script import run

    provider = _make_provider(VALID_SCRIPTS_JSON)
    cfg = _make_config()
    outline = _make_outline()

    run(outline, tmp_path, cfg, provider)

    scripts_file = tmp_path / "scene_scripts.json"
    assert scripts_file.exists()
    data = json.loads(scripts_file.read_text())
    assert data["topic"] == "derivatives"


def test_run_passes_outline_json_in_prompt(tmp_path):
    from mathmotion.stages.scene_script import run

    provider = _make_provider(VALID_SCRIPTS_JSON)
    cfg = _make_config()
    outline = _make_outline()

    run(outline, tmp_path, cfg, provider)

    call_args = provider.complete.call_args
    system_prompt = call_args.args[0]
    assert "Understanding Derivatives" in system_prompt


def test_run_passes_schema_to_provider(tmp_path):
    from mathmotion.stages.scene_script import run
    from mathmotion.schemas.script import AllSceneScripts

    provider = _make_provider(VALID_SCRIPTS_JSON)
    cfg = _make_config()
    outline = _make_outline()

    run(outline, tmp_path, cfg, provider)

    call_kwargs = provider.complete.call_args.kwargs
    assert call_kwargs["response_schema"] == AllSceneScripts.model_json_schema()


def test_run_raises_llm_error_after_max_retries(tmp_path):
    from mathmotion.stages.scene_script import run
    from mathmotion.utils.errors import LLMError

    provider = _make_provider("not valid json")
    cfg = _make_config(max_retries=0)
    outline = _make_outline()

    try:
        run(outline, tmp_path, cfg, provider)
        assert False, "Should have raised LLMError"
    except LLMError:
        pass


def test_run_retries_on_empty_narration(tmp_path):
    """Empty narration in a scene should trigger a retry."""
    from mathmotion.stages.scene_script import run

    empty_narration_json = json.dumps({
        "title": "Understanding Derivatives",
        "topic": "derivatives",
        "scenes": [
            {
                "id": "scene_1",
                "title": "Introduction",
                "narration": "   ",  # whitespace only — should fail validation
                "animation_description": {
                    "objects": [],
                    "sequence": [],
                    "notes": "",
                },
            }
        ],
    })

    good_resp = MagicMock()
    good_resp.content = VALID_SCRIPTS_JSON
    bad_resp = MagicMock()
    bad_resp.content = empty_narration_json

    provider = MagicMock()
    provider.complete.side_effect = [bad_resp, good_resp]
    cfg = _make_config(max_retries=1)
    outline = _make_outline()

    result = run(outline, tmp_path, cfg, provider)

    assert result.scenes[0].narration == "Today we learn about derivatives."
    assert provider.complete.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_stage_scene_script.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `mathmotion/stages/scene_script.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_stage_scene_script.py -v
```
Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add mathmotion/stages/scene_script.py tests/test_stage_scene_script.py
git commit -m "feat: implement scene_script stage (step 2 of 3)"
```

---

## Task 6: Implement `scene_code.py`

**Files:**
- Create: `mathmotion/stages/scene_code.py`
- Create: `tests/test_stage_scene_code.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stage_scene_code.py`:

```python
import json
from unittest.mock import MagicMock


MINIMAL_CODE = """\
from manim import Text, Scene

class Scene_Intro(Scene):
    def construct(self):
        t = Text("Hello")
        self.add(t)
        # WAIT:seg_1
        self.wait(1)
"""

VALID_SCENE_JSON = json.dumps({
    "id": "scene_1",
    "class_name": "Scene_Intro",
    "manim_code": MINIMAL_CODE,
    "narration_segments": [
        {"id": "seg_1", "text": "Today we learn about derivatives.", "cue_offset": 0.0}
    ],
})


def _make_outline():
    from mathmotion.schemas.script import TopicOutline, SceneOutlineItem
    return TopicOutline(
        title="Derivatives",
        topic="derivatives",
        level="undergraduate",
        scenes=[SceneOutlineItem(id="scene_1", title="Intro", purpose="Introduce", order=1)],
    )


def _make_scripts():
    from mathmotion.schemas.script import (
        AllSceneScripts, SceneScript, AnimationDescription,
        AnimationObject, AnimationStep,
    )
    return AllSceneScripts(
        title="Derivatives",
        topic="derivatives",
        scenes=[
            SceneScript(
                id="scene_1",
                title="Intro",
                narration="Today we learn about derivatives.",
                animation_description=AnimationDescription(
                    objects=[AnimationObject(id="title", type="Text", color="WHITE", initial_position="CENTER")],
                    sequence=[AnimationStep(action="FadeIn", target="title", timing="start", parameters={})],
                    notes="",
                ),
            )
        ],
    )


def _make_provider(content: str) -> MagicMock:
    provider = MagicMock()
    resp = MagicMock()
    resp.content = content
    provider.complete.return_value = resp
    return provider


def _make_config(max_retries: int = 1) -> MagicMock:
    cfg = MagicMock()
    cfg.llm.max_retries = max_retries
    cfg.llm.max_tokens = 4096
    cfg.llm.temperature = 0.2
    return cfg


def test_run_returns_generated_script(tmp_path):
    from mathmotion.stages.scene_code import run
    from mathmotion.schemas.script import GeneratedScript

    provider = _make_provider(VALID_SCENE_JSON)
    cfg = _make_config()

    result = run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)

    assert isinstance(result, GeneratedScript)
    assert result.title == "Derivatives"
    assert result.topic == "derivatives"
    assert len(result.scenes) == 1
    assert result.scenes[0].class_name == "Scene_Intro"


def test_run_writes_scene_files_and_narration_json(tmp_path):
    from mathmotion.stages.scene_code import run

    provider = _make_provider(VALID_SCENE_JSON)
    cfg = _make_config()

    run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)

    assert (tmp_path / "narration.json").exists()
    assert (tmp_path / "scenes" / "scene_1.py").exists()
    assert "Scene_Intro" in (tmp_path / "scenes" / "scene_1.py").read_text()


def test_run_does_not_write_files_if_any_scene_fails(tmp_path):
    from mathmotion.stages.scene_code import run
    from mathmotion.utils.errors import LLMError

    provider = _make_provider("not valid json")
    cfg = _make_config(max_retries=0)

    try:
        run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)
    except LLMError:
        pass

    assert not (tmp_path / "narration.json").exists()
    assert not (tmp_path / "scenes").exists()


def test_run_raises_llm_error_naming_failed_scenes(tmp_path):
    from mathmotion.stages.scene_code import run
    from mathmotion.utils.errors import LLMError

    provider = _make_provider("not valid json")
    cfg = _make_config(max_retries=0)

    try:
        run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)
        assert False, "Should have raised LLMError"
    except LLMError as e:
        assert "scene_1" in str(e)


def test_run_passes_outline_and_scene_script_to_provider(tmp_path):
    from mathmotion.stages.scene_code import run

    provider = _make_provider(VALID_SCENE_JSON)
    cfg = _make_config()

    run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)

    call_args = provider.complete.call_args
    system_prompt = call_args.args[0]
    assert "Derivatives" in system_prompt        # outline in prompt
    assert "Today we learn" in system_prompt     # scene narration in prompt


def test_run_retries_per_scene_on_invalid_json(tmp_path):
    """A scene that returns invalid JSON on attempt 1 should be retried and succeed on attempt 2."""
    from mathmotion.stages.scene_code import run
    from mathmotion.schemas.script import GeneratedScript

    bad_resp = MagicMock()
    bad_resp.content = "not valid json"
    good_resp = MagicMock()
    good_resp.content = VALID_SCENE_JSON

    provider = MagicMock()
    provider.complete.side_effect = [bad_resp, good_resp]

    cfg = _make_config(max_retries=1)
    result = run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)

    assert isinstance(result, GeneratedScript)
    assert result.scenes[0].class_name == "Scene_Intro"
    assert provider.complete.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_stage_scene_code.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `mathmotion/stages/scene_code.py`**

```python
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
    system_prompt = Path("prompts/scene_code.txt").read_text().format(
        outline_json=outline.model_dump_json(indent=2),
        scene_script_json=scene_script.model_dump_json(indent=2),
        schema_json=schema_json,
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_stage_scene_code.py -v
```
Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add mathmotion/stages/scene_code.py tests/test_stage_scene_code.py
git commit -m "feat: implement scene_code stage (step 3 of 3)"
```

---

## Task 7: Wire Up `pipeline.py`

**Files:**
- Modify: `mathmotion/pipeline.py`
- Modify: `tests/test_pipeline_upload.py`

- [ ] **Step 1: Run the existing pipeline tests to confirm they currently pass** (baseline)

```bash
uv run pytest tests/test_pipeline_upload.py -v
```
Expected: 2 PASSED (against the old pipeline). If they fail now, fix before continuing.

- [ ] **Step 2: Update `tests/test_pipeline_upload.py` to expect the new stage functions**

Both test functions reference `patch("mathmotion.pipeline.generate.run")` — one at line ~48 inside the `with` block, and one at line ~84. Replace **both** occurrences:

```python
# OLD (in both tests)
patch("mathmotion.pipeline.generate.run") as mock_generate,

# NEW (in both tests)
patch("mathmotion.pipeline.outline_stage.run") as mock_outline,
patch("mathmotion.pipeline.scene_script_stage.run") as mock_scripts,
patch("mathmotion.pipeline.scene_code_stage.run") as mock_code,
```

In `test_pipeline_skips_generate_when_script_provided`, also replace the assertion:
```python
# OLD
mock_generate.assert_not_called()

# NEW
mock_outline.assert_not_called()
mock_scripts.assert_not_called()
mock_code.assert_not_called()
```

The second test (`test_pipeline_writes_files_when_script_provided`) just needs the patch replaced — no assertion change needed since it doesn't assert on the generate mock directly.

- [ ] **Step 3: Run tests to confirm they now fail** (verifies TDD red state)

```bash
uv run pytest tests/test_pipeline_upload.py -v
```
Expected: FAILED with `AttributeError: <module 'mathmotion.pipeline'> does not have attribute 'outline_stage'` — this confirms the tests are correctly pointing at the new API.

- [ ] **Step 4: Update `mathmotion/pipeline.py`**

Replace the import line:
```python
# OLD
from mathmotion.stages import generate, render, repair, tts, compose

# NEW
from mathmotion.stages import outline as outline_stage
from mathmotion.stages import scene_script as scene_script_stage
from mathmotion.stages import scene_code as scene_code_stage
from mathmotion.stages import render, repair, tts, compose
```

Replace the generation block in `pipeline.run()`:
```python
# OLD
progress("Preparing script" if script is not None else "Generating script", 10)
provider = get_provider(config)
if script is None:
    script = generate.run(topic, job_dir, config, provider, level=level)
else:
    # ... bypass path unchanged ...

# NEW
progress("Preparing script" if script is not None else "Generating outline", 10)
provider = get_provider(config)
if script is None:
    outline_result = outline_stage.run(topic, job_dir, config, provider, level=level)
    progress("Writing scene scripts", 20)
    scripts_result = scene_script_stage.run(outline_result, job_dir, config, provider)
    progress("Generating scene code", 35)
    script = scene_code_stage.run(scripts_result, outline_result, job_dir, config, provider)
else:
    # bypass path: write files directly (unchanged)
    (job_dir / "scenes").mkdir(parents=True, exist_ok=True)
    for scene in script.scenes:
        (job_dir / "scenes" / f"{scene.id}.py").write_text(scene.manim_code)
    (job_dir / "narration.json").write_text(script.model_dump_json(indent=2))
```

Each `progress()` call fires immediately after the previous step completes and immediately before the next step starts: "Generating outline" at 10% announces the outline work is about to begin; "Writing scene scripts" at 20% fires after the outline finishes, before scene scripts start; "Generating scene code" at 35% fires after scene scripts finish, before code generation starts.

Update the remaining `progress()` calls to new percentages:
```python
progress("Synthesising audio", 45)         # was 30
# ...
progress("Injecting durations into scene code", 60)  # was 55
# ...
progress("Rendering animation", 70)        # was 65
# ...
progress("Composing final video", 88)      # was 85
```

- [ ] **Step 5: Run updated pipeline tests to verify they pass**

```bash
uv run pytest tests/test_pipeline_upload.py -v
```
Expected: 2 PASSED.

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
uv run pytest tests/ -v --ignore=tests/test_qwen3_engine.py
```
(Skip `test_qwen3_engine.py` only if it requires GPU — check first with `uv run pytest tests/test_qwen3_engine.py -v`.)
Expected: all tests PASSED.

- [ ] **Step 7: Commit**

```bash
git add mathmotion/pipeline.py tests/test_pipeline_upload.py
git commit -m "feat: wire three-stage generation pipeline in orchestrator"
```

---

## Task 8: Update `api/routes.py` Import

**Files:**
- Modify: `api/routes.py` line 86

- [ ] **Step 1: Run the API upload tests to confirm they currently pass** (baseline)

```bash
uv run pytest tests/test_api_upload.py -v
```
Expected: 4 PASSED.

- [ ] **Step 2: Update the import in `api/routes.py`**

Change line 86:
```python
# OLD
from mathmotion.stages.generate import validate_script

# NEW
from mathmotion.utils.validation import validate_script
```

- [ ] **Step 3: Run API upload tests to verify still passing**

```bash
uv run pytest tests/test_api_upload.py -v
```
Expected: 4 PASSED.

- [ ] **Step 4: Commit**

```bash
git add api/routes.py
git commit -m "fix: update validate_script import path in api/routes.py"
```

---

## Task 9: Delete `generate.py` and `system_prompt.txt`

**Files:**
- Delete: `mathmotion/stages/generate.py`
- Delete: `prompts/system_prompt.txt`

- [ ] **Step 1: Delete the old files**

```bash
git rm mathmotion/stages/generate.py prompts/system_prompt.txt
```

- [ ] **Step 2: Run the full test suite to verify nothing is broken**

```bash
uv run pytest tests/ -v
```
Expected: all tests PASSED. Any `ImportError` pointing to `generate` means a reference was missed — fix it before committing.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove superseded generate.py and system_prompt.txt"
```

---

## Final Verification

- [ ] **Run the complete test suite one final time**

```bash
uv run pytest tests/ -v
```
Expected: all tests PASSED, no references to `generate.py` or `system_prompt.txt`.

- [ ] **Verify disk artifacts are correct for a mocked run**

The pipeline should now produce `outline.json`, `scene_scripts.json`, `narration.json`, and `scenes/*.py` in the job directory. The existing tests already verify `narration.json` and scene file creation — the new stage tests verify `outline.json` and `scene_scripts.json`.
