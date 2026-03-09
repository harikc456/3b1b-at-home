def test_config_loads(monkeypatch):
    from mathmotion.utils.config import get_config
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.llm.model == "gemini/gemini-2.5-pro"
    assert cfg.llm.max_tokens == 8192
    assert cfg.llm.temperature == 0.2
    assert cfg.tts.engine in ("kokoro", "vibevoice", "qwen3")
    assert cfg.tts.kokoro.sample_rate == 24000
    assert cfg.server.port == 8000


def test_missing_env_does_not_raise():
    # With LiteLLM config, no env vars are required in config.yaml
    from mathmotion.utils.config import get_config
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.llm.model  # just has a model string
