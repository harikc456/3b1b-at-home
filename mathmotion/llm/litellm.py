import litellm
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError


class LiteLLMProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm

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

    @property
    def model_name(self) -> str:
        return self.cfg.model
