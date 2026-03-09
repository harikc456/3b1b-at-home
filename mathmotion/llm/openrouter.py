import httpx
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError


class OpenRouterProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm.openrouter

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 8192, temperature: float = 0.2,
                 response_schema: dict | None = None) -> LLMResponse:
        try:
            resp = httpx.post(
                f"{self.cfg.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.cfg.api_key}",
                    "HTTP-Referer": self.cfg.site_url,
                    "X-Title": self.cfg.site_name,
                },
                json={
                    "model": self.cfg.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                },
                timeout=120,
            )
        except httpx.RequestError as e:
            raise LLMError(f"OpenRouter network error: {e}") from e

        if resp.status_code != 200:
            raise LLMError(f"OpenRouter {resp.status_code}: {resp.text}")

        data = resp.json()
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.cfg.model),
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
        )

    @property
    def model_name(self) -> str:
        return self.cfg.model
