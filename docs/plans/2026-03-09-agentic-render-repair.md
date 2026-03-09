# Agentic Render Repair Loop — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reactive LLM-driven repair loop that catches Manim render failures, feeds the error + code back to the LLM for fixing, and retries — replacing the silent 3-retry-then-fallback behaviour in render.py.

**Architecture:** `pipeline.py` drives an outer repair loop: render all scenes → collect failures with their stderr → LLM fixes each broken scene file → re-render → repeat up to `repair_max_retries` times → fallback any still-failing scenes with the existing title card. A new `mathmotion/stages/repair.py` owns the LLM fix logic. `render.py` is refactored to expose `try_render_all()` that returns `(successes, failures)` instead of silently retrying internally.

**Tech Stack:** Python 3.13, Manim, LiteLLM (via existing `LiteLLMProvider`), Pydantic, pytest + unittest.mock

---

### Task 1: Add `repair_max_retries` to config

**Files:**
- Modify: `mathmotion/utils/config.py`
- Modify: `config.yaml`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

Open `tests/test_config.py` and add at the bottom:

```python
def test_llm_config_repair_max_retries_default():
    from mathmotion.utils.config import LLMConfig
    cfg = LLMConfig(model="gemini/gemini-2.5-pro")
    assert cfg.repair_max_retries == 3


def test_llm_config_repair_max_retries_custom():
    from mathmotion.utils.config import LLMConfig
    cfg = LLMConfig(model="gemini/gemini-2.5-pro", repair_max_retries=5)
    assert cfg.repair_max_retries == 5
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py::test_llm_config_repair_max_retries_default tests/test_config.py::test_llm_config_repair_max_retries_custom -v
```
Expected: FAIL with `ValidationError` or `AttributeError` — field doesn't exist yet.

**Step 3: Add field to `LLMConfig`**

In `mathmotion/utils/config.py`, add one line to `LLMConfig`:

```python
class LLMConfig(BaseModel):
    model: str
    models: list[str] = []
    max_tokens: int = 8192
    temperature: float = 0.2
    max_retries: int = 3
    repair_max_retries: int = 3   # ← add this
    timeout_seconds: int = 120
```

**Step 4: Add to `config.yaml`**

Under the `llm:` block in `config.yaml`, add after `max_retries`:

```yaml
  max_retries: 3
  repair_max_retries: 3
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```
Expected: all PASS.

**Step 6: Commit**

```bash
git add mathmotion/utils/config.py config.yaml tests/test_config.py
git commit -m "feat: add repair_max_retries to LLMConfig"
```

---

### Task 2: Add `json_mode` param to LLM provider interface

The repair call needs plain Python text back (not JSON), so `complete()` must be able to skip the `response_format` constraint.

**Files:**
- Modify: `mathmotion/llm/base.py`
- Modify: `mathmotion/llm/litellm.py`
- Test: `tests/test_litellm_provider.py`

**Step 1: Write the failing test**

Add to `tests/test_litellm_provider.py`:

```python
def test_complete_plain_text_skips_response_format():
    from mathmotion.llm.litellm import LiteLLMProvider

    config = _make_config()
    provider = LiteLLMProvider(config)

    mock_chunks = MagicMock()
    mock_chunks.__iter__ = MagicMock(return_value=iter([]))
    mock_resp = _make_litellm_response(content="def foo(): pass")

    with patch("mathmotion.llm.litellm.litellm.completion", return_value=mock_chunks) as mock_call, \
         patch("mathmotion.llm.litellm.litellm.stream_chunk_builder", return_value=mock_resp):
        result = provider.complete("sys", "user", json_mode=False)

    assert result.content == "def foo(): pass"
    call_kwargs = mock_call.call_args.kwargs
    assert "response_format" not in call_kwargs
```

**Step 2: Run to verify it fails**

```bash
pytest tests/test_litellm_provider.py::test_complete_plain_text_skips_response_format -v
```
Expected: FAIL — `complete()` doesn't accept `json_mode` yet.

**Step 3: Update `mathmotion/llm/base.py`**

Add `json_mode: bool = True` to the abstract method signature:

```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 8192, temperature: float = 0.2,
                 response_schema: dict | None = None,
                 json_mode: bool = True) -> LLMResponse: ...
```

**Step 4: Update `mathmotion/llm/litellm.py`**

Add `json_mode: bool = True` param and gate `response_format` on it:

```python
def complete(self, system_prompt: str, user_prompt: str,
             max_tokens: int = 8192, temperature: float = 0.2,
             response_schema: dict | None = None,
             json_mode: bool = True) -> LLMResponse:
    kwargs = dict(
        model=self.cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=self.cfg.timeout_seconds,
        stream=True,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        chunks = list(litellm.completion(**kwargs))
    except Exception as e:
        raise LLMError(f"LiteLLM error: {e}") from e

    response = litellm.stream_chunk_builder(chunks)
    return LLMResponse(
        content=response.choices[0].message.content,
        model=response.model,
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
    )
```

**Step 5: Run tests**

```bash
pytest tests/test_litellm_provider.py -v
```
Expected: all PASS.

**Step 6: Commit**

```bash
git add mathmotion/llm/base.py mathmotion/llm/litellm.py tests/test_litellm_provider.py
git commit -m "feat: add json_mode param to LLMProvider.complete"
```

---

### Task 3: Create `mathmotion/stages/repair.py`

**Files:**
- Create: `mathmotion/stages/repair.py`
- Create: `tests/test_repair.py`

**Step 1: Write failing tests**

Create `tests/test_repair.py`:

```python
import ast
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


VALID_CODE = '''\
from manim import Scene, Text, Write

class Scene_Test(Scene):
    def construct(self):
        t = Text("hello")
        self.play(Write(t))
        # WAIT:seg_1
        self.wait(1)
'''

BROKEN_CODE = '''\
from manim import Scene, Text, Write

class Scene_Test(Scene):
    def construct(self):
        t = Textt("hello")  # NameError - Textt doesn't exist
        self.play(Write(t))
        # WAIT:seg_1
        self.wait(1)
'''

STDERR = "NameError: name 'Textt' is not defined"


def _make_provider(response_content: str):
    provider = MagicMock()
    resp = MagicMock()
    resp.content = response_content
    provider.complete.return_value = resp
    return provider


def test_fix_scene_returns_fixed_code(tmp_path):
    from mathmotion.stages.repair import fix_scene

    scene_file = tmp_path / "scene_test.py"
    scene_file.write_text(BROKEN_CODE)

    provider = _make_provider(VALID_CODE)
    result = fix_scene(scene_file, STDERR, provider)

    assert result == VALID_CODE
    assert scene_file.read_text() == VALID_CODE  # file was overwritten


def test_fix_scene_prompt_contains_stderr_and_code(tmp_path):
    from mathmotion.stages.repair import fix_scene

    scene_file = tmp_path / "scene_test.py"
    scene_file.write_text(BROKEN_CODE)

    provider = _make_provider(VALID_CODE)
    fix_scene(scene_file, STDERR, provider)

    call_args = provider.complete.call_args
    user_prompt = call_args.args[1] if call_args.args else call_args.kwargs.get("user_prompt", "")
    assert STDERR in user_prompt
    assert BROKEN_CODE in user_prompt


def test_fix_scene_uses_plain_text_mode(tmp_path):
    from mathmotion.stages.repair import fix_scene

    scene_file = tmp_path / "scene_test.py"
    scene_file.write_text(BROKEN_CODE)

    provider = _make_provider(VALID_CODE)
    fix_scene(scene_file, STDERR, provider)

    call_kwargs = provider.complete.call_args.kwargs
    assert call_kwargs.get("json_mode") is False


def test_fix_scene_raises_on_invalid_python(tmp_path):
    from mathmotion.stages.repair import fix_scene
    from mathmotion.utils.errors import ValidationError

    scene_file = tmp_path / "scene_test.py"
    scene_file.write_text(BROKEN_CODE)

    provider = _make_provider("this is not python !!!! @@@")

    with pytest.raises(ValidationError, match="syntax"):
        fix_scene(scene_file, STDERR, provider)

    # File must NOT be overwritten with bad code
    assert scene_file.read_text() == BROKEN_CODE


def test_fix_scene_raises_on_forbidden_import(tmp_path):
    from mathmotion.stages.repair import fix_scene
    from mathmotion.utils.errors import ValidationError

    bad_fix = VALID_CODE.replace("from manim import", "import os\nfrom manim import")
    scene_file = tmp_path / "scene_test.py"
    scene_file.write_text(BROKEN_CODE)

    provider = _make_provider(bad_fix)

    with pytest.raises(ValidationError, match="forbidden"):
        fix_scene(scene_file, STDERR, provider)

    assert scene_file.read_text() == BROKEN_CODE


def test_fix_scene_strips_markdown_fences(tmp_path):
    from mathmotion.stages.repair import fix_scene

    scene_file = tmp_path / "scene_test.py"
    scene_file.write_text(BROKEN_CODE)

    fenced = f"```python\n{VALID_CODE}\n```"
    provider = _make_provider(fenced)
    result = fix_scene(scene_file, STDERR, provider)

    assert result == VALID_CODE
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_repair.py -v
```
Expected: all FAIL — module doesn't exist yet.

**Step 3: Implement `mathmotion/stages/repair.py`**

```python
import logging
import re
from pathlib import Path

from mathmotion.llm.base import LLMProvider
from mathmotion.utils.errors import LLMError, ValidationError
from mathmotion.utils.validation import check_forbidden_calls, check_forbidden_imports

logger = logging.getLogger(__name__)

REPAIR_SYSTEM_PROMPT = """\
You are an expert Manim animator. You will be given a broken Manim Python scene and the error it produced. Fix the code.

Rules:
- Preserve the exact class name
- Preserve all # WAIT:{seg_id} comments and the self.wait() lines that follow them
- Only import what you use (never use `from manim import *`)
- Forbidden imports: os, sys, subprocess, socket, urllib, requests, httpx
- Forbidden calls: open(), exec(), eval(), __import__()
- Return ONLY the raw Python code — no markdown fences, no explanation, nothing else
"""


def _strip_fences(code: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` wrappers if present."""
    code = code.strip()
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


def _validate_code(code: str) -> None:
    """Raises ValidationError if code has syntax errors or forbidden content."""
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        raise ValidationError(f"Fixed code has syntax error: {e}")

    bad = check_forbidden_imports(code)
    if bad:
        raise ValidationError(f"Fixed code has forbidden imports: {bad}")

    bad = check_forbidden_calls(code)
    if bad:
        raise ValidationError(f"Fixed code has forbidden calls: {bad}")


def fix_scene(scene_file: Path, stderr: str, provider: LLMProvider) -> str:
    """
    Ask the LLM to fix a broken Manim scene file.

    Reads scene_file, sends code + stderr to the LLM, validates the response,
    overwrites scene_file with the fixed code, and returns the fixed code.

    Raises ValidationError if the LLM's fix is itself invalid (scene_file is
    NOT overwritten in that case).
    """
    original_code = scene_file.read_text()
    user_prompt = (
        f"The following Manim scene failed to render.\n\n"
        f"ERROR:\n{stderr}\n\n"
        f"CODE:\n{original_code}\n\n"
        f"Fix the code."
    )

    logger.info(f"Requesting LLM repair for {scene_file.name}")
    try:
        resp = provider.complete(
            REPAIR_SYSTEM_PROMPT,
            user_prompt,
            json_mode=False,
        )
    except LLMError as e:
        raise ValidationError(f"LLM repair call failed: {e}") from e

    fixed_code = _strip_fences(resp.content)
    _validate_code(fixed_code)

    scene_file.write_text(fixed_code)
    logger.info(f"Repair written to {scene_file.name}")
    return fixed_code
```

**Step 4: Run tests**

```bash
pytest tests/test_repair.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add mathmotion/stages/repair.py tests/test_repair.py
git commit -m "feat: add repair stage with LLM-driven fix_scene"
```

---

### Task 4: Refactor `render.py` to expose `try_render_all`

Currently `run()` has an internal 3-retry loop that never involves the LLM. We extract a `try_render_all()` that does one render pass and returns `(successes, failures)`. The pipeline repair loop replaces the retries.

**Files:**
- Modify: `mathmotion/stages/render.py`
- Create: `tests/test_render.py`

**Step 1: Write failing tests**

Create `tests/test_render.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


def _make_scene(scene_id="scene_01", class_name="Scene_01"):
    scene = MagicMock()
    scene.id = scene_id
    scene.class_name = class_name
    scene.narration_segments = []
    return scene


def _make_config(quality="draft", timeout=30):
    cfg = MagicMock()
    cfg.manim.default_quality = quality
    cfg.manim.timeout_seconds = timeout
    cfg.manim.background_color = "#000000"
    return cfg


def test_try_render_all_success(tmp_path):
    from mathmotion.stages.render import try_render_all

    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    render_dir = tmp_path / "render"
    render_dir.mkdir()

    scene = _make_scene()
    config = _make_config()
    fake_mp4 = render_dir / "scene_01.mp4"

    with patch("mathmotion.stages.render._render", return_value=fake_mp4) as mock_render:
        successes, failures = try_render_all([scene], scenes_dir, render_dir, "draft", config)

    assert "scene_01" in successes
    assert successes["scene_01"] == fake_mp4
    assert failures == {}
    mock_render.assert_called_once()


def test_try_render_all_failure_captured(tmp_path):
    from mathmotion.stages.render import try_render_all
    from mathmotion.utils.errors import RenderError

    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    render_dir = tmp_path / "render"
    render_dir.mkdir()

    scene = _make_scene()
    config = _make_config()

    with patch("mathmotion.stages.render._render",
               side_effect=RenderError("scene_01", "NameError: boom")):
        successes, failures = try_render_all([scene], scenes_dir, render_dir, "draft", config)

    assert successes == {}
    assert "scene_01" in failures
    assert "NameError: boom" in failures["scene_01"]


def test_try_render_all_mixed(tmp_path):
    from mathmotion.stages.render import try_render_all
    from mathmotion.utils.errors import RenderError

    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    render_dir = tmp_path / "render"
    render_dir.mkdir()

    scene_ok = _make_scene("scene_01", "Scene_01")
    scene_fail = _make_scene("scene_02", "Scene_02")
    config = _make_config()
    fake_mp4 = render_dir / "scene_01.mp4"

    def side_effect(scene, *args, **kwargs):
        if scene.id == "scene_01":
            return fake_mp4
        raise RenderError("scene_02", "AttributeError: no method")

    with patch("mathmotion.stages.render._render", side_effect=side_effect):
        successes, failures = try_render_all(
            [scene_ok, scene_fail], scenes_dir, render_dir, "draft", config
        )

    assert "scene_01" in successes
    assert "scene_02" in failures
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_render.py -v
```
Expected: FAIL — `try_render_all` doesn't exist yet.

**Step 3: Add `try_render_all` to `render.py` and update `run()`**

In `mathmotion/stages/render.py`, add this function after `_fallback`:

```python
def try_render_all(
    scenes: list,
    scenes_dir: Path,
    render_dir: Path,
    quality: str,
    config,
) -> tuple[dict, dict]:
    """
    Attempt one render pass for each scene.

    Returns:
        successes: {scene_id: Path}
        failures:  {scene_id: stderr_str}
    """
    successes: dict = {}
    failures: dict = {}
    for scene in scenes:
        try:
            path = _render(scene, scenes_dir, render_dir, quality, config)
            successes[scene.id] = path
        except (RenderError, subprocess.TimeoutExpired) as e:
            stderr = e.stderr if isinstance(e, RenderError) else f"Render timed out for {scene.id}"
            logger.warning(f"Render failed for {scene.id}: {stderr[:200]}")
            failures[scene.id] = stderr
    return successes, failures
```

Then update `run()` to remove its internal retry loop and call `try_render_all` once (the repair loop in `pipeline.py` will handle retries):

```python
def run(script: GeneratedScript, job_dir: Path, config) -> dict[str, Path]:
    quality = config.manim.default_quality
    scenes_dir = job_dir / "scenes"
    render_dir = scenes_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rendering {len(script.scenes)} scene(s) with quality={quality!r}")
    successes, failures = try_render_all(script.scenes, scenes_dir, render_dir, quality, config)

    for scene in script.scenes:
        if scene.id in failures:
            logger.error(f"Scene {scene.id} failed — using fallback title card")
            successes[scene.id] = _fallback(scene, scenes_dir, render_dir, config)

    return successes
```

**Step 4: Run tests**

```bash
pytest tests/test_render.py tests/test_repair.py tests/test_config.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add mathmotion/stages/render.py tests/test_render.py
git commit -m "refactor: extract try_render_all from render.run, remove internal retry loop"
```

---

### Task 5: Wire the repair loop into `pipeline.py`

**Files:**
- Modify: `mathmotion/pipeline.py`
- Create: `tests/test_pipeline_repair.py`

**Step 1: Write failing tests**

Create `tests/test_pipeline_repair.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest


def _make_script(scene_ids=("scene_01", "scene_02")):
    script = MagicMock()
    scenes = []
    for sid in scene_ids:
        scene = MagicMock()
        scene.id = sid
        scene.class_name = f"Scene_{sid}"
        scene.narration_segments = []
        scenes.append(scene)
    script.scenes = scenes
    return script


def _make_config(repair_max_retries=2):
    cfg = MagicMock()
    cfg.manim.default_quality = "draft"
    cfg.manim.timeout_seconds = 30
    cfg.manim.background_color = "#000000"
    cfg.llm.repair_max_retries = repair_max_retries
    cfg.storage.jobs_dir = "/tmp/jobs"
    return cfg


def test_repair_loop_fixes_failing_scene(tmp_path):
    """When a scene fails on first render, repair fixes it, second render succeeds."""
    from mathmotion.stages.render import try_render_all
    from mathmotion.stages import repair

    script = _make_script(["scene_01"])
    config = _make_config(repair_max_retries=2)
    provider = MagicMock()

    fake_mp4 = tmp_path / "scene_01.mp4"
    render_calls = {"count": 0}

    def fake_try_render_all(scenes, *args, **kwargs):
        render_calls["count"] += 1
        if render_calls["count"] == 1:
            return {}, {"scene_01": "NameError: boom"}
        return {"scene_01": fake_mp4}, {}

    with patch("mathmotion.pipeline.render.try_render_all", side_effect=fake_try_render_all), \
         patch("mathmotion.pipeline.repair.fix_scene", return_value="fixed code") as mock_fix:
        from mathmotion.pipeline import _run_render_repair_loop
        results = _run_render_repair_loop(script, tmp_path, config, provider)

    assert results["scene_01"] == fake_mp4
    mock_fix.assert_called_once()


def test_repair_loop_fallback_after_exhausted(tmp_path):
    """When repair never fixes the scene, fallback is used."""
    from mathmotion.stages import repair

    script = _make_script(["scene_01"])
    config = _make_config(repair_max_retries=1)
    provider = MagicMock()

    fake_fallback = tmp_path / "scene_01_fallback.mp4"

    def always_fail(scenes, *args, **kwargs):
        return {}, {"scene_01": "always broken"}

    with patch("mathmotion.pipeline.render.try_render_all", side_effect=always_fail), \
         patch("mathmotion.pipeline.repair.fix_scene", return_value="fixed code"), \
         patch("mathmotion.pipeline.render._fallback", return_value=fake_fallback) as mock_fb:
        from mathmotion.pipeline import _run_render_repair_loop
        results = _run_render_repair_loop(script, tmp_path, config, provider)

    assert results["scene_01"] == fake_fallback
    mock_fb.assert_called_once()


def test_repair_loop_skips_fix_when_fix_invalid(tmp_path):
    """When fix_scene raises ValidationError, scene is carried to next round unchanged."""
    from mathmotion.stages import repair
    from mathmotion.utils.errors import ValidationError

    script = _make_script(["scene_01"])
    config = _make_config(repair_max_retries=2)
    provider = MagicMock()

    fake_fallback = tmp_path / "fallback.mp4"
    render_calls = {"count": 0}

    def always_fail(scenes, *args, **kwargs):
        render_calls["count"] += 1
        return {}, {"scene_01": "boom"}

    with patch("mathmotion.pipeline.render.try_render_all", side_effect=always_fail), \
         patch("mathmotion.pipeline.repair.fix_scene",
               side_effect=ValidationError("bad fix")), \
         patch("mathmotion.pipeline.render._fallback", return_value=fake_fallback):
        from mathmotion.pipeline import _run_render_repair_loop
        results = _run_render_repair_loop(script, tmp_path, config, provider)

    assert results["scene_01"] == fake_fallback
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_pipeline_repair.py -v
```
Expected: FAIL — `_run_render_repair_loop` doesn't exist yet.

**Step 3: Add `_run_render_repair_loop` to `pipeline.py`**

Add this import at the top of `mathmotion/pipeline.py`:

```python
from mathmotion.stages import repair
```

Then add `_run_render_repair_loop` as a new function (do NOT modify `run()` yet):

```python
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
            fixed = []
            for scene in remaining:
                try:
                    repair.fix_scene(
                        scenes_dir / f"{scene.id}.py",
                        failures[scene.id],
                        provider,
                    )
                    fixed.append(scene)
                except Exception as e:
                    logger.warning(f"LLM repair failed for {scene.id}: {e} — will retry as-is")
                    fixed.append(scene)
            remaining = fixed

    for scene in remaining:
        logger.error(f"Scene {scene.id} exhausted repair attempts — using fallback title card")
        results[scene.id] = _fallback(scene, scenes_dir, render_dir, config)

    return results
```

**Step 4: Run tests**

```bash
pytest tests/test_pipeline_repair.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add mathmotion/pipeline.py tests/test_pipeline_repair.py
git commit -m "feat: add _run_render_repair_loop to pipeline"
```

---

### Task 6: Swap `render.run()` for `_run_render_repair_loop` in `pipeline.run()`

**Files:**
- Modify: `mathmotion/pipeline.py`

**Step 1: Find the render call in `pipeline.run()`**

In `mathmotion/pipeline.py`, locate this block (around line 75):

```python
progress("Rendering animation", 65)
render.run(script, job_dir, config)
```

**Step 2: Replace with the repair loop call**

```python
progress("Rendering animation", 65)
_run_render_repair_loop(script, job_dir, config, provider)
```

Note: `provider` is already in scope — it was created at line 52 for the generate stage.

**Step 3: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: all PASS.

**Step 4: Commit**

```bash
git add mathmotion/pipeline.py
git commit -m "feat: use repair loop in pipeline, replacing silent render retries"
```

---

### Task 7: Smoke test end-to-end with a known broken scene

This is a manual verification step (no mocks) to confirm the loop works in practice.

**Step 1: Create a broken test scene**

```bash
mkdir -p /tmp/repair_smoke/scenes
cat > /tmp/repair_smoke/scenes/scene_01.py << 'EOF'
from manim import Scene, Text, Write

class Scene_SmokeTest(Scene):
    def construct(self):
        t = Textt("This is broken")  # Textt does not exist
        self.play(Write(t))
        # WAIT:seg_1
        self.wait(1)
EOF
```

**Step 2: Test `fix_scene` directly in a Python REPL**

```python
from pathlib import Path
from mathmotion.utils.config import get_config
from mathmotion.llm.factory import get_provider
from mathmotion.stages.repair import fix_scene

config = get_config()
provider = get_provider(config)
scene_file = Path("/tmp/repair_smoke/scenes/scene_01.py")
stderr = "NameError: name 'Textt' is not defined"

fixed = fix_scene(scene_file, stderr, provider)
print(fixed[:500])
```

Expected: LLM returns valid Python with `Text` instead of `Textt`. File is overwritten.

**Step 3: Verify the fixed code is parseable**

```python
import ast
ast.parse(Path("/tmp/repair_smoke/scenes/scene_01.py").read_text())
print("OK")
```

Expected: `OK` — no exception.
