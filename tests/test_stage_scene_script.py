import json
from unittest.mock import MagicMock


# LLM returns a single SceneScript JSON per scene (not AllSceneScripts)
VALID_SCENE_SCRIPT_JSON = json.dumps({
    "id": "scene_1",
    "title": "Introduction",
    "narration": "Today we learn about derivatives.",
    "animation_description": {
        "objects": [{"id": "title", "type": "Text", "color": "WHITE", "initial_position": "CENTER"}],
        "sequence": [{"action": "FadeIn", "target": "title", "timing": "start", "parameters": {}}],
        "notes": "",
    },
})


def _make_outline():
    from mathmotion.schemas.script import TopicOutline, SceneOutlineItem
    return TopicOutline(
        title="Understanding Derivatives",
        topic="derivatives",
        level="undergraduate",
        scenes=[SceneOutlineItem(id="scene_1", title="Introduction", purpose="Introduce", order=1)],
    )


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


def test_run_returns_all_scene_scripts(tmp_path):
    from mathmotion.stages.scene_script import run
    from mathmotion.schemas.script import AllSceneScripts

    provider = _make_provider(VALID_SCENE_SCRIPT_JSON)
    cfg = _make_config()
    outline = _make_outline()

    result = run(outline, tmp_path, cfg, provider)

    assert isinstance(result, AllSceneScripts)
    assert len(result.scenes) == 1
    assert result.scenes[0].narration == "Today we learn about derivatives."


def test_run_persists_scene_scripts_json(tmp_path):
    from mathmotion.stages.scene_script import run

    provider = _make_provider(VALID_SCENE_SCRIPT_JSON)
    cfg = _make_config()
    outline = _make_outline()

    run(outline, tmp_path, cfg, provider)

    scripts_file = tmp_path / "scene_scripts.json"
    assert scripts_file.exists()
    data = json.loads(scripts_file.read_text())
    assert data["topic"] == "derivatives"


def test_run_passes_outline_json_in_prompt(tmp_path):
    from mathmotion.stages.scene_script import run

    provider = _make_provider(VALID_SCENE_SCRIPT_JSON)
    cfg = _make_config()
    outline = _make_outline()

    run(outline, tmp_path, cfg, provider)

    call_args = provider.complete.call_args
    system_prompt = call_args.args[0]
    assert "Understanding Derivatives" in system_prompt


def test_run_does_not_pass_response_schema_to_provider(tmp_path):
    from mathmotion.stages.scene_script import run

    provider = _make_provider(VALID_SCENE_SCRIPT_JSON)
    cfg = _make_config()
    outline = _make_outline()

    run(outline, tmp_path, cfg, provider)

    call_kwargs = provider.complete.call_args.kwargs
    assert "response_schema" not in call_kwargs


def test_run_raises_llm_error_after_max_retries(tmp_path):
    from mathmotion.stages.scene_script import run
    from mathmotion.utils.errors import LLMError

    provider = _make_provider("not valid json")
    cfg = _make_config(max_retries=0)
    outline = _make_outline()

    try:
        run(outline, tmp_path, cfg, provider)
        assert False, "Should have raised LLMError"
    except LLMError:
        pass


def test_run_retries_on_empty_narration(tmp_path):
    """Empty narration in a scene should trigger a retry."""
    from mathmotion.stages.scene_script import run

    empty_narration_json = json.dumps({
        "id": "scene_1",
        "title": "Introduction",
        "narration": "   ",  # whitespace only — should fail validation
        "animation_description": {
            "objects": [],
            "sequence": [],
            "notes": "",
        },
    })

    good_resp = MagicMock()
    good_resp.content = VALID_SCENE_SCRIPT_JSON
    bad_resp = MagicMock()
    bad_resp.content = empty_narration_json

    provider = MagicMock()
    provider.complete.side_effect = [bad_resp, good_resp]
    cfg = _make_config(max_retries=1)
    outline = _make_outline()

    result = run(outline, tmp_path, cfg, provider)

    assert result.scenes[0].narration == "Today we learn about derivatives."
    assert provider.complete.call_count == 2
