import httpx
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError


class OllamaProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm.ollama

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 8192, temperature: float = 0.2) -> LLMResponse:
        try:
            resp = httpx.post(
                f"{self.cfg.base_url}/api/generate",
                json={
                    "model": self.cfg.model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=300,
            )
        except httpx.RequestError as e:
            raise LLMError(f"Ollama network error: {e}") from e

        if resp.status_code != 200:
            raise LLMError(f"Ollama {resp.status_code}: {resp.text}")

        data = resp.json()
        return LLMResponse(
            content=data["response"],
            model=data.get("model", self.cfg.model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    @property
    def model_name(self) -> str:
        return self.cfg.model
