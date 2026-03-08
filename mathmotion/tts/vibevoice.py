from pathlib import Path
from .base import TTSEngine
from mathmotion.utils.errors import TTSError
from mathmotion.utils.ffprobe import measure_duration

_synth = None


def _get_synth(voice: str, speed: float):
    global _synth
    if _synth is None:
        # vibevoice 0.0.1 — adjust import if API differs
        from vibevoice import Synthesizer
        _synth = Synthesizer(voice=voice, speed=speed)
    return _synth


class VibevoiceEngine(TTSEngine):
    def __init__(self, config):
        self.cfg = config.tts.vibevoice

    def synthesise(self, text: str, output_path: Path, voice: str = None, speed: float = 1.0) -> float:
        voice = voice or self.cfg.voice
        speed = speed or self.cfg.speed
        try:
            synth = _get_synth(voice, speed)
            wav_path = output_path.with_suffix(".wav")
            synth.save(text=text, output_path=str(wav_path), voice=voice)
            return measure_duration(wav_path)
        except Exception as e:
            raise TTSError(f"Vibevoice synthesis failed: {e}") from e

    def available_voices(self) -> list[str]:
        return list(self.cfg.available_voices)
