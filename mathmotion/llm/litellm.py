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
                response_format={"type": "json_object"},  # response_schema ignored; json_object sufficient for all current providers
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
