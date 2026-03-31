import json
from pathlib import Path


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


class VoiceoverScene:
    """Placeholder — completed in Task 2 once helpers are verified."""
    pass
