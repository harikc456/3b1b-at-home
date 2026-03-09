from .base import TTSEngine
from .kokoro import KokoroEngine
from .vibevoice import VibevoiceEngine
from .qwen3 import Qwen3Engine


def get_engine(config, engine_name: str = None) -> TTSEngine:
    name = engine_name or config.tts.engine
    match name:
        case "kokoro":    return KokoroEngine(config)
        case "vibevoice": return VibevoiceEngine(config)
        case "qwen3":     return Qwen3Engine(config)
        case n: raise ValueError(f"Unknown TTS engine: {n!r}. Valid: kokoro, vibevoice, qwen3")
