from typing import Optional
from pydantic import BaseModel


class NarrationSegment(BaseModel):
    id: str
    text: str
    cue_offset: float
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
