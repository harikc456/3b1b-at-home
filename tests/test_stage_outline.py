import json
from pathlib import Path
from unittest.mock import MagicMock, patch


VALID_OUTLINE_JSON = json.dumps({
    "title": "Understanding Derivatives",
    "topic": "derivatives",
    "level": "undergraduate",
    "scenes": [
        {"id": "scene_1", "title": "Introduction", "purpose": "Introduce concept", "order": 1},
        {"id": "scene_2", "title": "Formal Definition", "purpose": "Define derivative", "order": 2},
    ],
})


def _make_provider(content: str) -> MagicMock:
    provider = MagicMock()
    resp = MagicMock()
    resp.content = content
    provider.complete.return_value = resp
    return provider


def _make_config(max_retries: int = 1) -> MagicMock:
    cfg = MagicMock()
    cfg.llm.max_retries = max_retries
    cfg.llm.max_tokens = 4096
    cfg.llm.temperature = 0.2
    return cfg


def test_run_returns_topic_outline(tmp_path):
    from mathmotion.stages.outline import run
    from mathmotion.schemas.script import TopicOutline

    provider = _make_provider(VALID_OUTLINE_JSON)
    cfg = _make_config()

    result = run("derivatives", tmp_path, cfg, provider, level="undergraduate")

    assert isinstance(result, TopicOutline)
    assert result.topic == "derivatives"
    assert len(result.scenes) == 2
    assert result.scenes[0].id == "scene_1"


def test_run_persists_outline_json(tmp_path):
    from mathmotion.stages.outline import run

    provider = _make_provider(VALID_OUTLINE_JSON)
    cfg = _make_config()

    run("derivatives", tmp_path, cfg, provider)

    outline_file = tmp_path / "outline.json"
    assert outline_file.exists()
    data = json.loads(outline_file.read_text())
    assert data["topic"] == "derivatives"


def test_run_passes_schema_to_provider(tmp_path):
    from mathmotion.stages.outline import run
    from mathmotion.schemas.script import TopicOutline

    provider = _make_provider(VALID_OUTLINE_JSON)
    cfg = _make_config()

    run("derivatives", tmp_path, cfg, provider)

    call_kwargs = provider.complete.call_args.kwargs
    assert call_kwargs["response_schema"] == TopicOutline.model_json_schema()


def test_run_retries_on_invalid_json(tmp_path):
    from mathmotion.stages.outline import run

    provider = MagicMock()
    bad_resp = MagicMock()
    bad_resp.content = "not json"
    good_resp = MagicMock()
    good_resp.content = VALID_OUTLINE_JSON
    provider.complete.side_effect = [bad_resp, good_resp]

    cfg = _make_config(max_retries=1)
    result = run("derivatives", tmp_path, cfg, provider)

    assert result.topic == "derivatives"
    assert provider.complete.call_count == 2


def test_run_raises_llm_error_after_max_retries(tmp_path):
    from mathmotion.stages.outline import run
    from mathmotion.utils.errors import LLMError

    provider = _make_provider("not valid json at all")
    cfg = _make_config(max_retries=0)

    try:
        run("derivatives", tmp_path, cfg, provider)
        assert False, "Should have raised LLMError"
    except LLMError as e:
        assert "failed" in str(e).lower()
