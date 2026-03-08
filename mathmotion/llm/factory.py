from .base import LLMProvider
from .gemini import GeminiProvider
from .openrouter import OpenRouterProvider
from .ollama import OllamaProvider


def get_provider(config, provider_name: str = None) -> LLMProvider:
    name = provider_name or config.llm.provider
    match name:
        case "gemini":     return GeminiProvider(config)
        case "openrouter": return OpenRouterProvider(config)
        case "ollama":     return OllamaProvider(config)
        case p: raise ValueError(f"Unknown LLM provider: {p!r}. Valid: gemini, openrouter, ollama")
