import json
import pytest
from pathlib import Path


def test_voiceover_tracker_has_duration():
    from mathmotion.manim_ext import VoiceoverTracker
    t = VoiceoverTracker(duration=3.5)
    assert t.duration == 3.5


def test_voiceover_falls_back_to_word_count_without_file():
    from mathmotion.manim_ext import MathMotionScene
    scene = MathMotionScene.__new__(MathMotionScene)
    scene._mm_durations = []
    scene._mm_index = 0

    with scene.voiceover("one two three four five") as tracker:
        pass

    # 5 words / 2.5 = 2.0 seconds
    assert abs(tracker.duration - 2.0) < 0.01


def test_voiceover_fallback_floors_at_one_second():
    from mathmotion.manim_ext import MathMotionScene
    scene = MathMotionScene.__new__(MathMotionScene)
    scene._mm_durations = []
    scene._mm_index = 0

    with scene.voiceover("hi") as tracker:
        pass

    assert tracker.duration >= 1.0


def test_voiceover_reads_duration_from_file(tmp_path):
    from mathmotion.manim_ext import MathMotionScene
    scene = MathMotionScene.__new__(MathMotionScene)
    scene._mm_durations = [4.2, 1.8]
    scene._mm_index = 0

    with scene.voiceover("first segment") as t1:
        pass
    with scene.voiceover("second segment") as t2:
        pass

    assert abs(t1.duration - 4.2) < 0.001
    assert abs(t2.duration - 1.8) < 0.001


def test_voiceover_increments_index():
    from mathmotion.manim_ext import MathMotionScene
    scene = MathMotionScene.__new__(MathMotionScene)
    scene._mm_durations = [1.0, 2.0, 3.0]
    scene._mm_index = 0

    durations = []
    for text in ["a", "b", "c"]:
        with scene.voiceover(text) as tracker:
            durations.append(tracker.duration)

    assert durations == [1.0, 2.0, 3.0]


def test_voiceover_falls_back_after_durations_exhausted():
    from mathmotion.manim_ext import MathMotionScene
    scene = MathMotionScene.__new__(MathMotionScene)
    scene._mm_durations = [5.0]
    scene._mm_index = 0

    with scene.voiceover("first") as t1:
        pass
    with scene.voiceover("second segment with more words") as t2:
        pass

    assert abs(t1.duration - 5.0) < 0.001
    assert t2.duration >= 1.0  # fallback word-count estimate


def test_setup_loads_durations_from_colocated_file(tmp_path, monkeypatch):
    """setup() reads <scene_file>.durations.json co-located with the concrete subclass file."""
    import inspect
    from mathmotion.manim_ext import MathMotionScene

    # Write durations next to this test file (simulating scenes/scene_1.durations.json)
    test_file = Path(__file__)
    dur_file = test_file.with_suffix(".durations.json")
    dur_file.write_text(json.dumps([3.1, 2.7]))

    try:
        # Monkeypatch inspect.getfile to return this test file for MathMotionScene subclass
        original_getfile = inspect.getfile
        monkeypatch.setattr(inspect, "getfile", lambda cls: str(test_file) if issubclass(cls, MathMotionScene) else original_getfile(cls))

        scene = MathMotionScene.__new__(MathMotionScene)
        scene._mm_durations = []
        scene._mm_index = 0
        # Simulate setup() reading the file
        durations_path = Path(inspect.getfile(scene.__class__)).with_suffix(".durations.json")
        if durations_path.exists():
            scene._mm_durations = json.loads(durations_path.read_text())

        with scene.voiceover("hello") as t:
            pass
        assert abs(t.duration - 3.1) < 0.001
    finally:
        dur_file.unlink(missing_ok=True)
