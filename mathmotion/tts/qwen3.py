import logging
import torch
from pathlib import Path
import soundfile as sf

from .base import TTSEngine
from mathmotion.utils.errors import TTSError

logger = logging.getLogger(__name__)


class Qwen3Engine(TTSEngine):
    def __init__(self, config):
        self.cfg = config.tts.qwen3
        self._model = None

    def _load_model(self):
        from qwen_tts import Qwen3TTSModel
        attn = "flash_attention_2" if self.cfg.use_flash_attention else "eager"
        self._model = Qwen3TTSModel.from_pretrained(
            self.cfg.model_id,
            device_map=self.cfg.device,
            dtype=torch.bfloat16,
            attn_implementation=attn,
        )

    def synthesise(self, text: str, output_path: Path, voice: str = None, speed: float = 1.0) -> float:
        if self._model is None:
            self._load_model()

        if speed != 1.0:
            logger.warning("Qwen3Engine does not support speed control; speed=%s is ignored", speed)

        speaker = voice or self.cfg.voice
        try:
            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language=self.cfg.language,
                speaker=speaker,
                instruct=self.cfg.instruct,
            )
            wav_path = output_path.with_suffix(".wav")
            sf.write(str(wav_path), wavs[0], sr)
        except Exception as e:
            raise TTSError(f"Qwen3 synthesis failed: {e}") from e

        return len(wavs[0]) / sr

    def available_voices(self) -> list[str]:
        return list(self.cfg.available_voices)
