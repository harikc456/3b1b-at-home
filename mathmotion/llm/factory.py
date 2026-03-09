from .base import LLMProvider
from .litellm import LiteLLMProvider


def get_provider(config, provider_name: str = None) -> LLMProvider:
    return LiteLLMProvider(config)
