from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 8192, temperature: float = 0.2,
                 response_schema: dict | None = None) -> LLMResponse: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...
