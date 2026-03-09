# Qwen3 TTS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `Qwen3Engine` as a new TTS backend using `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` and make it the default engine.

**Architecture:** Follow the existing `TTSEngine` pattern — new `Qwen3Config` Pydantic model in `config.py`, new `mathmotion/tts/qwen3.py` engine, one-line addition to the factory, and a three-way fix in the TTS stage. The model is lazy-loaded on first `synthesise()` call.

**Tech Stack:** `qwen_tts` (PyPI), `soundfile`, `torch`, existing `TTSEngine` ABC.

---

### Task 1: Add `Qwen3Config` to config and update `TTSConfig`

**Files:**
- Modify: `mathmotion/utils/config.py`
- Modify: `tests/test_config.py`

**Step 1: Update `test_config.py` to expect `qwen3` as a valid engine**

Replace line 9 in `tests/test_config.py`:
```python
# Before
assert cfg.tts.engine in ("kokoro", "vibevoice")

# After
assert cfg.tts.engine in ("kokoro", "vibevoice", "qwen3")
```

Also add a new assertion after line 10:
```python
assert cfg.tts.qwen3.voice == "Vivian"
```

**Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_config.py::test_config_loads -v
```

Expected: FAIL — `TTSConfig` has no `qwen3` attribute.

**Step 3: Add `Qwen3Config` and update `TTSConfig` in `mathmotion/utils/config.py`**

Add this class after `VibevoiceConfig` (around line 56):

```python
class Qwen3Config(BaseModel):
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    device: str = "cuda:0"
    use_flash_attention: bool = False
    voice: str = "Vivian"
    speed: float = 1.0          # accepted by interface, ignored in synthesis
    instruct: str = ""
    language: str = "English"
    available_voices: list[str] = ["Vivian"]
```

Update `TTSConfig` (around line 59) to add the `qwen3` field and change the default engine:

```python
class TTSConfig(BaseModel):
    engine: str = "qwen3"       # changed from "kokoro"
    kokoro: KokoroConfig
    vibevoice: VibevoiceConfig
    qwen3: Qwen3Config = Qwen3Config()
```

**Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_config.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add mathmotion/utils/config.py tests/test_config.py
git commit -m "feat: add Qwen3Config and update TTSConfig default engine"
```

---

### Task 2: Create the Qwen3Engine

**Files:**
- Create: `mathmotion/tts/qwen3.py`
- Create: `tests/test_qwen3_engine.py`

**Step 1: Write the failing test**

Create `tests/test_qwen3_engine.py`:

```python
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


def _make_config():
    """Build a minimal Config-like object for Qwen3Engine."""
    qwen3_cfg = types.SimpleNamespace(
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device="cpu",
        use_flash_attention=False,
        voice="Vivian",
        speed=1.0,
        instruct="",
        language="English",
        available_voices=["Vivian"],
    )
    tts_cfg = types.SimpleNamespace(qwen3=qwen3_cfg)
    return types.SimpleNamespace(tts=tts_cfg)


def test_qwen3_engine_synthesise_returns_duration(tmp_path):
    sample_rate = 24000
    audio = np.zeros(sample_rate * 2, dtype=np.float32)   # 2 seconds

    mock_model = MagicMock()
    mock_model.generate_custom_voice.return_value = ([audio], sample_rate)

    mock_cls = MagicMock(return_value=mock_model)
    mock_cls.from_pretrained.return_value = mock_model

    with patch.dict("sys.modules", {"qwen_tts": MagicMock(Qwen3TTSModel=mock_cls)}):
        from mathmotion.tts.qwen3 import Qwen3Engine
        engine = Qwen3Engine(_make_config())
        out = tmp_path / "seg"
        duration = engine.synthesise("Hello world", out)

    assert abs(duration - 2.0) < 0.01
    assert out.with_suffix(".wav").exists()


def test_qwen3_engine_uses_voice_param(tmp_path):
    sample_rate = 24000
    audio = np.zeros(sample_rate, dtype=np.float32)

    mock_model = MagicMock()
    mock_model.generate_custom_voice.return_value = ([audio], sample_rate)

    mock_cls = MagicMock(return_value=mock_model)
    mock_cls.from_pretrained.return_value = mock_model

    with patch.dict("sys.modules", {"qwen_tts": MagicMock(Qwen3TTSModel=mock_cls)}):
        from mathmotion.tts.qwen3 import Qwen3Engine
        engine = Qwen3Engine(_make_config())
        engine.synthesise("Hi", tmp_path / "seg", voice="CustomSpeaker")

    call_kwargs = mock_model.generate_custom_voice.call_args.kwargs
    assert call_kwargs["speaker"] == "CustomSpeaker"


def test_qwen3_engine_available_voices():
    with patch.dict("sys.modules", {"qwen_tts": MagicMock()}):
        from mathmotion.tts.qwen3 import Qwen3Engine
        engine = Qwen3Engine(_make_config())
    assert engine.available_voices() == ["Vivian"]
```

**Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_qwen3_engine.py -v
```

Expected: FAIL — `mathmotion.tts.qwen3` does not exist.

**Step 3: Create `mathmotion/tts/qwen3.py`**

```python
import torch
from pathlib import Path
import soundfile as sf

from .base import TTSEngine
from mathmotion.utils.errors import TTSError


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

        speaker = voice or self.cfg.voice
        try:
            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language=self.cfg.language,
                speaker=speaker,
                instruct=self.cfg.instruct,
            )
        except Exception as e:
            raise TTSError(f"Qwen3 synthesis failed: {e}") from e

        wav_path = output_path.with_suffix(".wav")
        sf.write(str(wav_path), wavs[0], sr)
        return len(wavs[0]) / sr

    def available_voices(self) -> list[str]:
        return list(self.cfg.available_voices)
```

**Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_qwen3_engine.py -v
```

Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add mathmotion/tts/qwen3.py tests/test_qwen3_engine.py
git commit -m "feat: add Qwen3Engine TTS implementation"
```

---

### Task 3: Update factory and TTS stage

**Files:**
- Modify: `mathmotion/tts/factory.py`
- Modify: `mathmotion/stages/tts.py`

**Step 1: Update `factory.py`**

Replace the contents of `mathmotion/tts/factory.py`:

```python
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
```

**Step 2: Fix the engine config selection in `mathmotion/stages/tts.py`**

Replace lines 38-40 (the two-way config selection) with a three-way match:

```python
# Before
tts_cfg = config.tts.kokoro if config.tts.engine == "kokoro" else config.tts.vibevoice

# After
match config.tts.engine:
    case "kokoro":    tts_cfg = config.tts.kokoro
    case "vibevoice": tts_cfg = config.tts.vibevoice
    case "qwen3":     tts_cfg = config.tts.qwen3
    case n:           raise ValueError(f"Unknown TTS engine: {n!r}")
```

**Step 3: Run all existing tests**

```bash
uv run pytest -v
```

Expected: All tests PASS (the stage fix has no unit tests since stage tests require integration setup, but existing tests should not regress).

**Step 4: Commit**

```bash
git add mathmotion/tts/factory.py mathmotion/stages/tts.py
git commit -m "feat: register Qwen3Engine in factory and fix stage engine config selection"
```

---

### Task 4: Update `config.yaml` and `pyproject.toml`

**Files:**
- Modify: `config.yaml`
- Modify: `pyproject.toml`

**Step 1: Update `config.yaml`**

Change `engine: kokoro` → `engine: qwen3` and add the `qwen3:` block after the `vibevoice:` block:

```yaml
tts:
  engine: qwen3                # kokoro | vibevoice | qwen3

  kokoro:
    lang_code: a
    voice: af_heart
    speed: 1.0
    sample_rate: 24000
    available_voices:
      - af_heart
      - af_sky
      - af_bella
      - bm_lewis
      - bm_george

  vibevoice:
    voice: neutral_female
    speed: 1.0
    available_voices:
      - neutral_female
      - neutral_male
      - expressive_female

  qwen3:
    model_id: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
    device: cuda:0
    use_flash_attention: false
    voice: Vivian
    instruct: ""
    language: English
    available_voices:
      - Vivian
```

**Step 2: Add `qwen-tts` to `pyproject.toml` dependencies**

Add `"qwen-tts>=0.1"` to the `dependencies` list in `pyproject.toml`.

**Step 3: Sync dependencies**

```bash
uv sync
```

Expected: `qwen-tts` installs successfully.

**Step 4: Run full test suite**

```bash
uv run pytest -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add config.yaml pyproject.toml uv.lock
git commit -m "feat: set qwen3 as default TTS engine and add qwen-tts dependency"
```
