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


def test_qwen3_engine_raises_tts_error_on_failure(tmp_path):
    mock_model = MagicMock()
    mock_model.generate_custom_voice.side_effect = RuntimeError("model exploded")

    mock_cls = MagicMock()
    mock_cls.from_pretrained.return_value = mock_model

    from mathmotion.utils.errors import TTSError
    with patch.dict("sys.modules", {"qwen_tts": MagicMock(Qwen3TTSModel=mock_cls)}):
        from mathmotion.tts.qwen3 import Qwen3Engine
        engine = Qwen3Engine(_make_config())
        with pytest.raises(TTSError, match="Qwen3 synthesis failed"):
            engine.synthesise("Hello", tmp_path / "seg")
