from .base import TTSEngine
from .kokoro import KokoroEngine
from .vibevoice import VibevoiceEngine


def get_engine(config, engine_name: str = None) -> TTSEngine:
    name = engine_name or config.tts.engine
    match name:
        case "kokoro":    return KokoroEngine(config)
        case "vibevoice": return VibevoiceEngine(config)
        case n: raise ValueError(f"Unknown TTS engine: {n!r}. Valid: kokoro, vibevoice")
