# Qwen3 TTS Support Design

**Date:** 2026-03-09
**Status:** Approved

## Summary

Add `Qwen3Engine` as a new TTS backend using `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` via the `qwen_tts` package. Make it the default engine.

## Model

- **Package:** `qwen_tts`
- **Checkpoint:** `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- **Method:** `model.generate_custom_voice(text, language, speaker, instruct)`
- **Returns:** `(wavs, sr)` — list of numpy arrays + sample rate

## Config Changes

### `config.py`

New `Qwen3Config` Pydantic model:

```python
class Qwen3Config(BaseModel):
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    device: str = "cuda:0"
    use_flash_attention: bool = False
    speaker: str = "Vivian"
    instruct: str = ""
    language: str = "English"
    available_voices: list[str] = ["Vivian"]
```

`TTSConfig` gains a `qwen3: Qwen3Config` field. Default engine changes from `"kokoro"` to `"qwen3"`.

### `config.yaml`

- Change `engine: kokoro` → `engine: qwen3`
- Add `qwen3:` block with defaults

## New File: `mathmotion/tts/qwen3.py`

`Qwen3Engine(TTSEngine)`:
- Lazy-loads model on first `synthesise()` call
- `attn_implementation` set to `"flash_attention_2"` if `use_flash_attention=True`, else `"eager"`
- `synthesise(text, output_path, voice=None, speed=1.0)`:
  - `voice` → `speaker` (falls back to `cfg.speaker`)
  - `speed` → ignored
  - writes WAV via `soundfile.write()`
  - returns `len(wavs[0]) / sr`
- `available_voices()` → `list(cfg.available_voices)`

## Factory Changes (`factory.py`)

Add `case "qwen3": return Qwen3Engine(config)`. Update error message to include `qwen3`.

## Stage Fix (`stages/tts.py`)

Replace the hardcoded two-way check for voice/speed config:
```python
tts_cfg = config.tts.kokoro if config.tts.engine == "kokoro" else config.tts.vibevoice
```
With a three-way check (or small helper) that handles `qwen3` → `config.tts.qwen3`.

## Out of Scope

- Speed control (not supported by this model)
- Multiple instruct presets per voice
- CPU fallback configuration
