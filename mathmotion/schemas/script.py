from typing import Optional
from pydantic import BaseModel


class NarrationSegment(BaseModel):
    id: str
    text: str
    actual_duration: Optional[float] = None
    audio_path: Optional[str] = None


class Scene(BaseModel):
    id: str
    class_name: str
    manim_code: str
    narration_segments: list[NarrationSegment]


class GeneratedScript(BaseModel):
    title: str
    topic: str
    scenes: list[Scene]


# ── Step 1 schemas ────────────────────────────────────────────────────────────

class SceneOutlineItem(BaseModel):
    id: str
    title: str
    purpose: str
    order: int


class TopicOutline(BaseModel):
    title: str
    topic: str
    level: str
    scenes: list[SceneOutlineItem]


# ── Step 2 schemas ────────────────────────────────────────────────────────────

class AnimationObject(BaseModel):
    id: str
    type: str
    color: str
    initial_position: str


class AnimationStep(BaseModel):
    action: str
    target: str
    timing: str
    parameters: dict[str, int | float | str]


class AnimationDescription(BaseModel):
    objects: list[AnimationObject]
    sequence: list[AnimationStep]
    notes: str = ""


class SceneScript(BaseModel):
    id: str
    title: str
    narration: str
    animation_description: AnimationDescription


class AllSceneScripts(BaseModel):
    title: str
    topic: str
    scenes: list[SceneScript]
