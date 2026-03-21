import pytest
from pydantic import ValidationError


def test_topic_outline_valid():
    from mathmotion.schemas.script import TopicOutline, SceneOutlineItem
    outline = TopicOutline(
        title="Derivatives",
        topic="derivatives",
        level="undergraduate",
        scenes=[SceneOutlineItem(id="scene_1", title="Intro", purpose="Introduce concept", order=1)],
    )
    assert outline.level == "undergraduate"
    assert outline.scenes[0].id == "scene_1"


def test_topic_outline_rejects_wrong_type_for_scenes():
    from mathmotion.schemas.script import TopicOutline
    with pytest.raises(ValidationError):
        TopicOutline(title="X", topic="x", level="undergraduate", scenes="not-a-list")


def test_all_scene_scripts_valid():
    from mathmotion.schemas.script import (
        AllSceneScripts, SceneScript, AnimationDescription,
        AnimationObject, AnimationStep,
    )
    scripts = AllSceneScripts(
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
    assert scripts.scenes[0].narration == "Today we learn about derivatives."


def test_animation_step_parameters_typed():
    from mathmotion.schemas.script import AnimationStep
    step = AnimationStep(
        action="MoveTo",
        target="circle_1",
        timing="after_narration_segment_1",
        parameters={"scale": 2, "color": "BLUE"},
    )
    assert step.parameters["scale"] == 2
