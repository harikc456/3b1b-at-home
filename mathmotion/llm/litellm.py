import logging

import litellm
from litellm import supports_response_schema
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError

logger = logging.getLogger(__name__)


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
        )
        if json_mode:
            schema_supported = bool(response_schema and supports_response_schema(model=self.cfg.model))
            if schema_supported:
                logger.debug(f"Using json_schema response_format for {self.cfg.model}")
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": response_schema, "strict": True},
                }
            else:
                logger.debug(
                    f"Using json_object response_format for {self.cfg.model} "
                    f"(response_schema={'passed' if response_schema else 'not passed'}, "
                    f"supports_response_schema={supports_response_schema(model=self.cfg.model)})"
                )
                kwargs["response_format"] = {"type": "json_object"}
        try:
            response = litellm.completion(**kwargs)
        except Exception as e:
            raise LLMError(f"LiteLLM error: {e}") from e

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    @property
    def model_name(self) -> str:
        return self.cfg.model
