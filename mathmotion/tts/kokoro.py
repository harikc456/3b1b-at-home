from pathlib import Path
from .base import TTSEngine
from mathmotion.utils.errors import TTSError

_pipeline = None


def _get_pipeline(lang_code: str):
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code=lang_code)
    return _pipeline


class KokoroEngine(TTSEngine):
    def __init__(self, config):
        self.cfg = config.tts.kokoro

    def synthesise(self, text: str, output_path: Path, voice: str = None, speed: float = 1.0) -> float:
        import numpy as np
        import soundfile as sf

        pipeline = _get_pipeline(self.cfg.lang_code)
        voice = voice or self.cfg.voice
        try:
            chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
            audio = np.concatenate(chunks)
            wav_path = output_path.with_suffix(".wav")
            sf.write(str(wav_path), audio, self.cfg.sample_rate)
            return len(audio) / self.cfg.sample_rate
        except Exception as e:
            raise TTSError(f"Kokoro synthesis failed: {e}") from e

    def available_voices(self) -> list[str]:
        return list(self.cfg.available_voices)
