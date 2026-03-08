def test_config_loads(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    from mathmotion.utils.config import get_config
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.llm.provider in ("gemini", "openrouter", "ollama")
    assert cfg.llm.gemini.api_key == "test-gemini-key"
    assert cfg.tts.engine in ("kokoro", "vibevoice")
    assert cfg.tts.kokoro.sample_rate == 24000
    assert cfg.server.port == 8000


def test_missing_env_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from mathmotion.utils.config import get_config, ConfigError
    get_config.cache_clear()
    try:
        get_config()
        assert False, "Should have raised ConfigError"
    except ConfigError:
        pass
    finally:
        get_config.cache_clear()
