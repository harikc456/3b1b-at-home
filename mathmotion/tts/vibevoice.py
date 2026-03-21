from pathlib import Path
from .base import TTSEngine
from mathmotion.utils.errors import TTSError
from mathmotion.utils.ffprobe import measure_duration

_model = None
_processor = None


def _load(model_path: str, device: str, ddpm_steps: int):
    global _model, _processor
    if _model is None:
        import torch
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        _processor = VibeVoiceProcessor.from_pretrained(model_path)

        load_dtype = torch.float32 if device in ("mps", "cpu") else torch.bfloat16
        attn_impl = "flash_attention_2" if device == "cuda" else "sdpa"
        try:
            _model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=device if device != "mps" else None,
                attn_implementation=attn_impl,
            )
        except Exception:
            _model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=device if device != "mps" else None,
                attn_implementation="sdpa",
            )
        if device == "mps":
            _model.to("mps")
        _model.eval()
        _model.set_ddpm_inference_steps(num_steps=ddpm_steps)
    return _model, _processor


class VibevoiceEngine(TTSEngine):
    def __init__(self, config):
        self.cfg = config.tts.vibevoice

    def synthesise(self, text: str, output_path: Path, voice: str = None, speed: float = 1.0) -> float:
        import torch
        voice = voice or self.cfg.voice
        voice_sample = self.cfg.voice_samples.get(voice)
        if not voice_sample:
            raise TTSError(
                f"No voice sample path configured for voice {voice!r}. "
                "Add it under tts.vibevoice.voice_samples in config.yaml."
            )
        try:
            model, processor = _load(
                self.cfg.model_path,
                self.cfg.device,
                self.cfg.ddpm_inference_steps,
            )
            script = f"Speaker 1: {text}"
            inputs = processor(
                text=[script],
                voice_samples=[[voice_sample]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            device = self.cfg.device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=self.cfg.cfg_scale,
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    show_progress_bar=False,
                )

            wav_path = output_path.with_suffix(".wav")
            processor.save_audio(outputs.speech_outputs[0], output_path=str(wav_path))
            return measure_duration(wav_path)
        except TTSError:
            raise
        except Exception as e:
            raise TTSError(f"Vibevoice synthesis failed: {e}") from e

    def available_voices(self) -> list[str]:
        return list(self.cfg.available_voices)
