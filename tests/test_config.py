def test_config_loads():
    from mathmotion.utils.config import get_config
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.llm.model == "gemini/gemini-2.0-flash"
    assert cfg.llm.max_tokens == 16000
    assert cfg.llm.temperature == 0.0
    assert cfg.tts.engine in ("kokoro", "vibevoice", "qwen3")
    assert cfg.tts.kokoro.sample_rate == 24000
    assert cfg.server.port == 8000


def test_missing_env_does_not_raise():
    # With LiteLLM config, no env vars are required in config.yaml
    from mathmotion.utils.config import get_config
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.llm.model  # just has a model string


def test_llm_config_repair_max_retries_default():
    from mathmotion.utils.config import LLMConfig
    cfg = LLMConfig(model="gemini/gemini-2.5-pro")
    assert cfg.repair_max_retries == 3


def test_llm_config_repair_max_retries_custom():
    from mathmotion.utils.config import LLMConfig
    cfg = LLMConfig(model="gemini/gemini-2.5-pro", repair_max_retries=5)
    assert cfg.repair_max_retries == 5
