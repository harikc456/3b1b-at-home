import json
from contextlib import contextmanager
from pathlib import Path

from manim import Scene, config as manim_config


def _load_audio_map(scene_file: Path) -> dict:
    """Load {seg_id: {audio_path, duration}} from the sidecar next to scene_file."""
    sidecar = scene_file.with_name(scene_file.stem + "_voiceover.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return {}


def _remaining_wait(duration: float, elapsed: float, frame_rate: float) -> float:
    """Return how long to wait after the voiceover context exits, or 0 if negligible."""
    remaining = duration - elapsed
    if remaining > 1 / frame_rate:
        return remaining
    return 0.0


class VoiceoverTracker:
    """Passed to the voiceover context body so LLM code can read audio duration."""
    def __init__(self, duration: float) -> None:
        self.duration = duration


class VoiceoverScene(Scene):
    """Manim Scene subclass that embeds TTS audio at render time via add_sound()."""

    def setup(self) -> None:
        super().setup()
        self._audio_map = _load_audio_map(Path(manim_config.input_file))

    @contextmanager
    def voiceover(self, seg_id: str):
        """Context manager that plays audio for seg_id and waits for its duration.

        Usage::

            with self.voiceover("seg_scene01_001") as tracker:
                self.play(Write(text), run_time=tracker.duration)
            # auto-waits for remaining audio after context exits
        """
        entry = self._audio_map.get(seg_id, {})
        audio_path = entry.get("audio_path")
        duration = entry.get("duration", 0.0)

        start_t = self.renderer.time
        if audio_path and Path(audio_path).exists():
            self.add_sound(audio_path)

        yield VoiceoverTracker(duration)

        elapsed = self.renderer.time - start_t
        remaining = _remaining_wait(duration, elapsed, manim_config.frame_rate)
        if remaining > 0:
            self.wait(remaining)
