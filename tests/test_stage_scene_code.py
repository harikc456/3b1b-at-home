import pytest
from unittest.mock import MagicMock


MINIMAL_VOICEOVER_CODE = """\
from manim import *
from mathmotion.manim_ext import MathMotionScene

class Scene_Intro(MathMotionScene):
    def construct(self):
        t = Text("Hello")
        with self.voiceover("Today we learn about derivatives.") as tracker:
            self.play(Write(t), run_time=tracker.duration)
"""

FENCED_VOICEOVER_CODE = "```python\n" + MINIMAL_VOICEOVER_CODE + "```"

TWO_SEGMENT_CODE = """\
from manim import *
from mathmotion.manim_ext import MathMotionScene

class Scene_Intro(MathMotionScene):
    def construct(self):
        t = Text("Hello")
        with self.voiceover("First segment narration.") as tracker:
            self.play(Write(t), run_time=tracker.duration)
        with self.voiceover("Second segment narration.") as tracker:
            self.play(FadeIn(t), run_time=tracker.duration)
"""


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
    cfg.llm.model = "gemini-2.0-flash"
    cfg.llm.max_parallel_scenes = 4
    return cfg


# ── Parser tests ─────────────────────────────────────────────────────────────

def test_parse_code_to_scene_extracts_class_name():
    from mathmotion.stages.scene_code import _parse_code_to_scene
    scene = _parse_code_to_scene("scene_1", MINIMAL_VOICEOVER_CODE)
    assert scene.class_name == "Scene_Intro"


def test_parse_code_to_scene_extracts_narration_segments():
    from mathmotion.stages.scene_code import _parse_code_to_scene
    scene = _parse_code_to_scene("scene_1", TWO_SEGMENT_CODE)
    assert len(scene.narration_segments) == 2
    assert scene.narration_segments[0].id == "seg_0"
    assert scene.narration_segments[0].text == "First segment narration."
    assert scene.narration_segments[1].id == "seg_1"
    assert scene.narration_segments[1].text == "Second segment narration."


def test_parse_code_to_scene_strips_markdown_fences():
    from mathmotion.stages.scene_code import _parse_code_to_scene
    scene = _parse_code_to_scene("scene_1", FENCED_VOICEOVER_CODE)
    assert scene.class_name == "Scene_Intro"
    assert "```" not in scene.manim_code


def test_parse_code_to_scene_sets_scene_id():
    from mathmotion.stages.scene_code import _parse_code_to_scene
    scene = _parse_code_to_scene("scene_42", MINIMAL_VOICEOVER_CODE)
    assert scene.id == "scene_42"


def test_parse_code_to_scene_raises_on_missing_class():
    from mathmotion.stages.scene_code import _parse_code_to_scene
    with pytest.raises(ValueError, match="No Scene_ class"):
        _parse_code_to_scene("scene_1", "from manim import *\n# no class here\n")


# ── run() integration tests ───────────────────────────────────────────────────

def test_run_returns_generated_script(tmp_path):
    from mathmotion.stages.scene_code import run
    from mathmotion.schemas.script import GeneratedScript

    provider = _make_provider(MINIMAL_VOICEOVER_CODE)
    result = run(_make_scripts(), _make_outline(), tmp_path, _make_config(), provider)

    assert isinstance(result, GeneratedScript)
    assert result.title == "Derivatives"
    assert len(result.scenes) == 1
    assert result.scenes[0].class_name == "Scene_Intro"


def test_run_writes_scene_files_and_narration_json(tmp_path):
    from mathmotion.stages.scene_code import run

    run(_make_scripts(), _make_outline(), tmp_path, _make_config(), _make_provider(MINIMAL_VOICEOVER_CODE))

    assert (tmp_path / "narration.json").exists()
    assert (tmp_path / "scenes" / "scene_1.py").exists()
    assert "Scene_Intro" in (tmp_path / "scenes" / "scene_1.py").read_text()


def test_run_does_not_write_files_if_any_scene_fails(tmp_path):
    from mathmotion.stages.scene_code import run
    from mathmotion.utils.errors import LLMError

    provider = _make_provider("not valid python with no Scene_ class")
    with pytest.raises(LLMError):
        run(_make_scripts(), _make_outline(), tmp_path, _make_config(max_retries=0), provider)

    assert not (tmp_path / "narration.json").exists()


def test_run_raises_llm_error_naming_failed_scenes(tmp_path):
    from mathmotion.stages.scene_code import run
    from mathmotion.utils.errors import LLMError

    provider = _make_provider("no Scene_ class here")
    with pytest.raises(LLMError, match="scene_1"):
        run(_make_scripts(), _make_outline(), tmp_path, _make_config(max_retries=0), provider)


def test_run_passes_outline_and_scene_script_to_provider(tmp_path):
    from mathmotion.stages.scene_code import run

    provider = _make_provider(MINIMAL_VOICEOVER_CODE)
    run(_make_scripts(), _make_outline(), tmp_path, _make_config(), provider)

    system_prompt = provider.complete.call_args.args[0]
    assert "Derivatives" in system_prompt
    assert "Today we learn" in system_prompt


def test_run_calls_provider_with_json_mode_false(tmp_path):
    from mathmotion.stages.scene_code import run

    provider = _make_provider(MINIMAL_VOICEOVER_CODE)
    run(_make_scripts(), _make_outline(), tmp_path, _make_config(), provider)

    call_kwargs = provider.complete.call_args.kwargs
    assert call_kwargs.get("json_mode") is False


def test_run_retries_on_parse_error(tmp_path):
    from mathmotion.stages.scene_code import run
    from mathmotion.schemas.script import GeneratedScript

    bad_resp = MagicMock()
    bad_resp.content = "no Scene_ class"
    good_resp = MagicMock()
    good_resp.content = MINIMAL_VOICEOVER_CODE

    provider = MagicMock()
    provider.complete.side_effect = [bad_resp, good_resp]

    result = run(_make_scripts(), _make_outline(), tmp_path, _make_config(max_retries=1), provider)

    assert isinstance(result, GeneratedScript)
    assert provider.complete.call_count == 2


def test_run_writes_error_log_on_retry(tmp_path):
    """Failed attempts are written to scene_code_errors.jsonl."""
    import json as _json
    from mathmotion.stages.scene_code import run

    bad_resp = MagicMock()
    bad_resp.content = "no Scene_ class"
    good_resp = MagicMock()
    good_resp.content = MINIMAL_VOICEOVER_CODE

    provider = MagicMock()
    provider.complete.side_effect = [bad_resp, good_resp]

    run(_make_scripts(), _make_outline(), tmp_path, _make_config(max_retries=1), provider)

    errors_file = tmp_path / "scene_code_errors.jsonl"
    assert errors_file.exists()
    records = [_json.loads(line) for line in errors_file.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["scene_id"] == "scene_1"
    assert records[0]["attempt"] == 0
    assert "Parse error" in records[0]["error"]
    assert records[0]["raw_response"] == "no Scene_ class"


def test_run_scene_id_comes_from_input_not_llm(tmp_path):
    """LLM returning Scene_Wrong class does not affect scene.id."""
    from mathmotion.stages.scene_code import run

    wrong_class_code = MINIMAL_VOICEOVER_CODE.replace("Scene_Intro", "Scene_Wrong")
    provider = _make_provider(wrong_class_code)
    result = run(_make_scripts(), _make_outline(), tmp_path, _make_config(), provider)

    assert result.scenes[0].id == "scene_1"
    assert (tmp_path / "scenes" / "scene_1.py").exists()
