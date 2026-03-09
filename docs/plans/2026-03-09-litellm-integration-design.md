# LiteLLM Integration Design

**Date:** 2026-03-09
**Status:** Approved

## Goal

Replace three custom LLM provider implementations (Gemini, OpenRouter, Ollama) with a single `LiteLLMProvider` backed by the `litellm` library. This enables any provider supported by LiteLLM via a single model string in config.

## Decisions

- **Model naming:** LiteLLM-style model strings (e.g. `gemini/gemini-2.5-pro`, `openrouter/qwen/qwen3-next-80b`, `ollama/ministral-3:14b`)
- **API keys:** Resolved from environment variables via `.env` — LiteLLM picks them up automatically (no per-provider key fields in config)
- **Config shape:** Single `model` string replaces three provider sub-sections

## File Changes

| File | Action |
|------|--------|
| `mathmotion/llm/litellm.py` | **Create** — `LiteLLMProvider` implementation |
| `mathmotion/llm/factory.py` | **Simplify** — always returns `LiteLLMProvider` |
| `mathmotion/llm/gemini.py` | **Delete** |
| `mathmotion/llm/openrouter.py` | **Delete** |
| `mathmotion/llm/ollama.py` | **Delete** |
| `mathmotion/llm/base.py` | **Unchanged** |
| `mathmotion/utils/config.py` | **Simplify** — drop `GeminiConfig`, `OpenRouterConfig`, `OllamaConfig`; `LLMConfig` keeps `model: str` + shared params |
| `config.yaml` | **Simplify** — replace three provider sub-sections with `model: gemini/gemini-2.5-pro` |
| `pyproject.toml` | **Update** — add `litellm`, remove `google-generativeai` |

## Unchanged

- `mathmotion/llm/base.py` (`LLMProvider`, `LLMResponse`)
- `mathmotion/stages/generate.py`
- `mathmotion/pipeline.py`
- All API routes and TTS system

## `LiteLLMProvider` sketch

```python
import litellm
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError

class LiteLLMProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm

    def complete(self, system_prompt, user_prompt,
                 max_tokens=8192, temperature=0.2,
                 response_schema=None) -> LLMResponse:
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

## New `config.yaml` LLM section

```yaml
llm:
  model: gemini/gemini-2.5-pro   # any litellm model string
  max_tokens: 8192
  temperature: 0.2
  max_retries: 3
  timeout_seconds: 120
```
