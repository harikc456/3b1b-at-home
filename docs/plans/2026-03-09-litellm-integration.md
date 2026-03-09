# LiteLLM Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace three custom LLM provider classes (Gemini, OpenRouter, Ollama) with a single `LiteLLMProvider` backed by the `litellm` library, using LiteLLM-style model strings in config.

**Architecture:** A single `LiteLLMProvider` class implements the existing `LLMProvider` abstract base. Config simplifies to one `model` string (e.g. `gemini/gemini-2.5-pro`) plus shared params. API keys are resolved from `.env` automatically by LiteLLM — no per-provider config sections needed.

**Tech Stack:** `litellm`, `pydantic`, `pytest`, `uv`

---

### Task 1: Add `litellm` dependency, remove `google-generativeai`

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update dependencies**

In `pyproject.toml`, replace `"google-generativeai>=0.8",` with `"litellm>=1.0",`:

```toml
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "httpx>=0.27",
    "python-dotenv>=1.0",
    "filelock>=3.13",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.29",
    "manim>=0.18.0",
    "kokoro>=0.9.2",
    "soundfile>=0.12",
    "numpy>=1.26",
    "spacy>=3.8.11",
    "en-core-web-sm",
    "qwen-tts>=0.1,<1.0",
    "litellm>=1.0",
]
```

**Step 2: Sync the environment**

```bash
uv sync
```

Expected: resolves and installs `litellm`, removes `google-generativeai`.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: replace google-generativeai with litellm"
```

---

### Task 2: Simplify `LLMConfig` in `config.py`

**Files:**
- Modify: `mathmotion/utils/config.py`

**Step 1: Write failing test**

In `tests/test_config.py`, replace the existing `test_config_loads` and `test_missing_env_raises` with:

```python
def test_config_loads(monkeypatch):
    from mathmotion.utils.config import get_config
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.llm.model == "gemini/gemini-2.5-pro"
    assert cfg.llm.max_tokens == 8192
    assert cfg.llm.temperature == 0.2
    assert cfg.tts.engine in ("kokoro", "vibevoice", "qwen3")
    assert cfg.tts.kokoro.sample_rate == 24000
    assert cfg.server.port == 8000


def test_missing_env_does_not_raise():
    # With LiteLLM config, no env vars are required in config.yaml
    from mathmotion.utils.config import get_config
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.llm.model  # just has a model string
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `cfg.llm` has no `model` attribute, still has old shape.

**Step 3: Update `LLMConfig` in `config.py`**

Replace the three provider config classes and `LLMConfig`:

```python
# DELETE: GeminiConfig, OpenRouterConfig, OllamaConfig classes

class LLMConfig(BaseModel):
    model: str
    max_tokens: int = 8192
    temperature: float = 0.2
    max_retries: int = 3
    timeout_seconds: int = 120
```

The full updated `config.py` imports section and models (keep all TTS/Manim/etc. classes unchanged):

```python
class LLMConfig(BaseModel):
    model: str
    max_tokens: int = 8192
    temperature: float = 0.2
    max_retries: int = 3
    timeout_seconds: int = 120
```

**Step 4: Update `config.yaml` LLM section**

Replace the entire `llm:` block:

```yaml
llm:
  model: gemini/gemini-2.5-pro   # any litellm model string
  max_tokens: 8192
  temperature: 0.2
  max_retries: 3
  timeout_seconds: 120
```

Delete the `gemini:`, `openrouter:`, and `ollama:` sub-sections entirely.

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: PASS.

**Step 6: Commit**

```bash
git add mathmotion/utils/config.py config.yaml tests/test_config.py
git commit -m "refactor: simplify LLMConfig to single model string for litellm"
```

---

### Task 3: Create `LiteLLMProvider`

**Files:**
- Create: `mathmotion/llm/litellm.py`
- Create: `tests/test_litellm_provider.py`

**Step 1: Write failing test**

Create `tests/test_litellm_provider.py`:

```python
from unittest.mock import MagicMock, patch


def _make_config(model="gemini/gemini-2.5-pro", timeout=120):
    cfg = MagicMock()
    cfg.llm.model = model
    cfg.llm.timeout_seconds = timeout
    return cfg


def _make_litellm_response(content='{"key": "value"}', model="gemini/gemini-2.5-pro",
                            prompt_tokens=10, completion_tokens=20):
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.model = model
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def test_complete_returns_llm_response():
    from mathmotion.llm.litellm import LiteLLMProvider
    from mathmotion.llm.base import LLMResponse

    config = _make_config()
    provider = LiteLLMProvider(config)

    mock_resp = _make_litellm_response()
    with patch("mathmotion.llm.litellm.litellm.completion", return_value=mock_resp) as mock_call:
        result = provider.complete("sys", "user", max_tokens=100, temperature=0.5)

    assert isinstance(result, LLMResponse)
    assert result.content == '{"key": "value"}'
    assert result.model == "gemini/gemini-2.5-pro"
    assert result.input_tokens == 10
    assert result.output_tokens == 20

    mock_call.assert_called_once_with(
        model="gemini/gemini-2.5-pro",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user",   "content": "user"},
        ],
        max_tokens=100,
        temperature=0.5,
        response_format={"type": "json_object"},
        timeout=120,
    )


def test_complete_wraps_exception_in_llm_error():
    from mathmotion.llm.litellm import LiteLLMProvider
    from mathmotion.utils.errors import LLMError

    config = _make_config()
    provider = LiteLLMProvider(config)

    with patch("mathmotion.llm.litellm.litellm.completion", side_effect=RuntimeError("boom")):
        try:
            provider.complete("sys", "user")
            assert False, "Should have raised LLMError"
        except LLMError as e:
            assert "boom" in str(e)


def test_model_name_property():
    from mathmotion.llm.litellm import LiteLLMProvider

    config = _make_config(model="ollama/ministral-3:14b")
    provider = LiteLLMProvider(config)
    assert provider.model_name == "ollama/ministral-3:14b"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_litellm_provider.py -v
```

Expected: FAIL — `mathmotion.llm.litellm` does not exist.

**Step 3: Create `mathmotion/llm/litellm.py`**

```python
import litellm
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError


class LiteLLMProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 8192, temperature: float = 0.2,
                 response_schema: dict | None = None) -> LLMResponse:
        try:
            response = litellm.completion(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=self.cfg.timeout_seconds,
            )
        except Exception as e:
            raise LLMError(f"LiteLLM error: {e}") from e
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    @property
    def model_name(self) -> str:
        return self.cfg.model
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_litellm_provider.py -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add mathmotion/llm/litellm.py tests/test_litellm_provider.py
git commit -m "feat: add LiteLLMProvider"
```

---

### Task 4: Simplify factory, delete old provider files

**Files:**
- Modify: `mathmotion/llm/factory.py`
- Delete: `mathmotion/llm/gemini.py`
- Delete: `mathmotion/llm/openrouter.py`
- Delete: `mathmotion/llm/ollama.py`

**Step 1: Update `factory.py`**

Replace the entire file with:

```python
from .base import LLMProvider
from .litellm import LiteLLMProvider


def get_provider(config, provider_name: str = None) -> LLMProvider:
    return LiteLLMProvider(config)
```

Note: `provider_name` is kept in the signature for API compatibility but is no longer used — LiteLLM determines the provider from the model string in config.

**Step 2: Delete old provider files**

```bash
git rm mathmotion/llm/gemini.py mathmotion/llm/openrouter.py mathmotion/llm/ollama.py
```

**Step 3: Run full test suite**

```bash
pytest -v
```

Expected: all tests PASS. No import errors for removed files.

**Step 4: Commit**

```bash
git add mathmotion/llm/factory.py
git commit -m "refactor: simplify factory to always use LiteLLMProvider, remove old provider files"
```

---

### Task 5: Verify end-to-end and run full suite

**Step 1: Run full test suite**

```bash
pytest -v
```

Expected: all tests PASS.

**Step 2: Verify config loads correctly**

```bash
python -c "
from mathmotion.utils.config import get_config
cfg = get_config()
print('model:', cfg.llm.model)
print('max_tokens:', cfg.llm.max_tokens)
"
```

Expected output:
```
model: gemini/gemini-2.5-pro
max_tokens: 8192
```

**Step 3: Verify factory returns LiteLLMProvider**

```bash
python -c "
from mathmotion.utils.config import get_config
from mathmotion.llm.factory import get_provider
cfg = get_config()
p = get_provider(cfg)
print(type(p).__name__, p.model_name)
"
```

Expected output:
```
LiteLLMProvider gemini/gemini-2.5-pro
```
