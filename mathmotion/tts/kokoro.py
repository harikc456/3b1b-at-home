from pathlib import Path
from .base import TTSEngine
from mathmotion.utils.errors import TTSError


class KokoroEngine(TTSEngine):
    def __init__(self, config):
        self.cfg = config.tts.kokoro
        self._pipeline = None

    def synthesise(self, text: str, output_path: Path, voice: str = None, speed: float = 1.0) -> float:
        import numpy as np
        import soundfile as sf

        if self._pipeline is None:
            print("[Kokoro] Loading KPipeline...", flush=True)
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code=self.cfg.lang_code, repo_id='hexgrad/Kokoro-82M')
            print("[Kokoro] KPipeline loaded.", flush=True)
        pipeline = self._pipeline
        voice = voice or self.cfg.voice
        try:
            print(f"Generating audio for {text} with voice={voice} and speed={speed}")
            chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
            audio = np.concatenate(chunks)
            wav_path = output_path.with_suffix(".wav")
            print(f"Saving audio to {wav_path}")
            sf.write(str(wav_path), audio, self.cfg.sample_rate)
            return len(audio) / self.cfg.sample_rate
        except Exception as e:
            raise TTSError(f"Kokoro synthesis failed: {e}") from e

    def available_voices(self) -> list[str]:
        return list(self.cfg.available_voices)
