import json
from unittest.mock import MagicMock


MINIMAL_CODE = """\
from manim import Text, Scene

class Scene_Intro(Scene):
    def construct(self):
        t = Text("Hello")
        self.add(t)
        # WAIT:seg_1
        self.wait(1)
"""

VALID_SCENE_JSON = json.dumps({
    "id": "scene_1",
    "class_name": "Scene_Intro",
    "manim_code": MINIMAL_CODE,
    "narration_segments": [
        {"id": "seg_1", "text": "Today we learn about derivatives.", "cue_offset": 0.0}
    ],
})


def _make_outline():
    from mathmotion.schemas.script import TopicOutline, SceneOutlineItem
    return TopicOutline(
        title="Derivatives",
        topic="derivatives",
        level="undergraduate",
        scenes=[SceneOutlineItem(id="scene_1", title="Intro", purpose="Introduce", order=1)],
    )


def _make_scripts():
    from mathmotion.schemas.script import (
        AllSceneScripts, SceneScript, AnimationDescription,
        AnimationObject, AnimationStep,
    )
    return AllSceneScripts(
        title="Derivatives",
        topic="derivatives",
        scenes=[
            SceneScript(
                id="scene_1",
                title="Intro",
                narration="Today we learn about derivatives.",
                animation_description=AnimationDescription(
                    objects=[AnimationObject(id="title", type="Text", color="WHITE", initial_position="CENTER")],
                    sequence=[AnimationStep(action="FadeIn", target="title", timing="start", parameters={})],
                    notes="",
                ),
            )
        ],
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


def test_run_returns_generated_script(tmp_path):
    from mathmotion.stages.scene_code import run
    from mathmotion.schemas.script import GeneratedScript

    provider = _make_provider(VALID_SCENE_JSON)
    cfg = _make_config()

    result = run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)

    assert isinstance(result, GeneratedScript)
    assert result.title == "Derivatives"
    assert result.topic == "derivatives"
    assert len(result.scenes) == 1
    assert result.scenes[0].class_name == "Scene_Intro"


def test_run_writes_scene_files_and_narration_json(tmp_path):
    from mathmotion.stages.scene_code import run

    provider = _make_provider(VALID_SCENE_JSON)
    cfg = _make_config()

    run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)

    assert (tmp_path / "narration.json").exists()
    assert (tmp_path / "scenes" / "scene_1.py").exists()
    assert "Scene_Intro" in (tmp_path / "scenes" / "scene_1.py").read_text()


def test_run_does_not_write_files_if_any_scene_fails(tmp_path):
    import pytest
    from mathmotion.stages.scene_code import run
    from mathmotion.utils.errors import LLMError

    provider = _make_provider("not valid json")
    cfg = _make_config(max_retries=0)

    with pytest.raises(LLMError):
        run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)

    assert not (tmp_path / "narration.json").exists()
    assert not (tmp_path / "scenes").exists()


def test_run_raises_llm_error_naming_failed_scenes(tmp_path):
    from mathmotion.stages.scene_code import run
    from mathmotion.utils.errors import LLMError

    provider = _make_provider("not valid json")
    cfg = _make_config(max_retries=0)

    try:
        run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)
        assert False, "Should have raised LLMError"
    except LLMError as e:
        assert "scene_1" in str(e)


def test_run_passes_outline_and_scene_script_to_provider(tmp_path):
    from mathmotion.stages.scene_code import run

    provider = _make_provider(VALID_SCENE_JSON)
    cfg = _make_config()

    run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)

    call_args = provider.complete.call_args
    system_prompt = call_args.args[0]
    assert "Derivatives" in system_prompt        # outline in prompt
    assert "Today we learn" in system_prompt     # scene narration in prompt


def test_run_retries_per_scene_on_invalid_json(tmp_path):
    """A scene that returns invalid JSON on attempt 1 should be retried and succeed on attempt 2."""
    from mathmotion.stages.scene_code import run
    from mathmotion.schemas.script import GeneratedScript

    bad_resp = MagicMock()
    bad_resp.content = "not valid json"
    good_resp = MagicMock()
    good_resp.content = VALID_SCENE_JSON

    provider = MagicMock()
    provider.complete.side_effect = [bad_resp, good_resp]

    cfg = _make_config(max_retries=1)
    result = run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)

    assert isinstance(result, GeneratedScript)
    assert result.scenes[0].class_name == "Scene_Intro"
    assert provider.complete.call_count == 2


def test_run_passes_schema_to_provider(tmp_path):
    from mathmotion.stages.scene_code import run
    from mathmotion.schemas.script import Scene

    provider = _make_provider(VALID_SCENE_JSON)
    cfg = _make_config()

    run(_make_scripts(), _make_outline(), tmp_path, cfg, provider)

    call_kwargs = provider.complete.call_args.kwargs
    assert call_kwargs["response_schema"] == Scene.model_json_schema()
