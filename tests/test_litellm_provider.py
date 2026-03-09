from unittest.mock import MagicMock, patch


def _make_config(model="gemini/gemini-2.5-pro", timeout=120):
    cfg = MagicMock()
    cfg.llm.model = model
    cfg.llm.timeout_seconds = timeout
    return cfg


def _make_litellm_response(content='{"key": "value"}', model="gemini/gemini-2.5-pro",
                            prompt_tokens=10, completion_tokens=20):
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.model = model
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def test_complete_returns_llm_response():
    from mathmotion.llm.litellm import LiteLLMProvider
    from mathmotion.llm.base import LLMResponse

    config = _make_config()
    provider = LiteLLMProvider(config)

    mock_resp = _make_litellm_response()
    with patch("mathmotion.llm.litellm.litellm.completion", return_value=mock_resp) as mock_call:
        result = provider.complete("sys", "user", max_tokens=100, temperature=0.5)

    assert isinstance(result, LLMResponse)
    assert result.content == '{"key": "value"}'
    assert result.model == "gemini/gemini-2.5-pro"
    assert result.input_tokens == 10
    assert result.output_tokens == 20

    mock_call.assert_called_once_with(
        model="gemini/gemini-2.5-pro",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user",   "content": "user"},
        ],
        max_tokens=100,
        temperature=0.5,
        response_format={"type": "json_object"},
        timeout=120,
    )


def test_complete_wraps_exception_in_llm_error():
    from mathmotion.llm.litellm import LiteLLMProvider
    from mathmotion.utils.errors import LLMError

    config = _make_config()
    provider = LiteLLMProvider(config)

    with patch("mathmotion.llm.litellm.litellm.completion", side_effect=RuntimeError("boom")):
        try:
            provider.complete("sys", "user")
            assert False, "Should have raised LLMError"
        except LLMError as e:
            assert "boom" in str(e)


def test_model_name_property():
    from mathmotion.llm.litellm import LiteLLMProvider

    config = _make_config(model="ollama/ministral-3:14b")
    provider = LiteLLMProvider(config)
    assert provider.model_name == "ollama/ministral-3:14b"


def test_complete_plain_text_skips_response_format():
    from mathmotion.llm.litellm import LiteLLMProvider
    from unittest.mock import MagicMock, patch

    config = _make_config()
    provider = LiteLLMProvider(config)

    mock_chunks = MagicMock()
    mock_chunks.__iter__ = MagicMock(return_value=iter([]))
    mock_resp = _make_litellm_response(content="def foo(): pass")

    with patch("mathmotion.llm.litellm.litellm.completion", return_value=mock_chunks) as mock_call, \
         patch("mathmotion.llm.litellm.litellm.stream_chunk_builder", return_value=mock_resp):
        result = provider.complete("sys", "user", json_mode=False)

    assert result.content == "def foo(): pass"
    call_kwargs = mock_call.call_args.kwargs
    assert "response_format" not in call_kwargs
