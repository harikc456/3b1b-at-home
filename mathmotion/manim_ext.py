import inspect
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from manim import Scene


@dataclass
class VoiceoverTracker:
    duration: float


class MathMotionScene(Scene):
    def setup(self):
        super().setup()
        self._mm_durations: list[float] = []
        self._mm_index: int = 0
        scene_file = Path(inspect.getfile(self.__class__))
        durations_path = scene_file.with_suffix(".durations.json")
        if durations_path.exists():
            self._mm_durations = json.loads(durations_path.read_text())

    @contextmanager
    def voiceover(self, text: str):
        if self._mm_index < len(self._mm_durations):
            duration = self._mm_durations[self._mm_index]
        else:
            duration = max(1.0, len(text.split()) / 2.5)
        self._mm_index += 1
        yield VoiceoverTracker(duration=duration)
