import json
import pytest


def test_voiceover_tracker_has_duration():
    from mathmotion.manim_ext import VoiceoverTracker
    t = VoiceoverTracker(duration=3.5)
    assert t.duration == 3.5


def test_voiceover_falls_back_to_word_count_without_file(monkeypatch):
    monkeypatch.delenv("MATHMOTION_DURATIONS_FILE", raising=False)
    from mathmotion.manim_ext import MathMotionScene
    scene = MathMotionScene.__new__(MathMotionScene)
    scene._mm_durations = []
    scene._mm_index = 0

    with scene.voiceover("one two three four five") as tracker:
        pass

    # 5 words / 2.5 = 2.0 seconds
    assert abs(tracker.duration - 2.0) < 0.01


def test_voiceover_fallback_floors_at_one_second(monkeypatch):
    monkeypatch.delenv("MATHMOTION_DURATIONS_FILE", raising=False)
    from mathmotion.manim_ext import MathMotionScene
    scene = MathMotionScene.__new__(MathMotionScene)
    scene._mm_durations = []
    scene._mm_index = 0

    with scene.voiceover("hi") as tracker:
        pass

    assert tracker.duration >= 1.0


def test_voiceover_reads_duration_from_file(tmp_path, monkeypatch):
    dur_file = tmp_path / "durations.json"
    dur_file.write_text(json.dumps([4.2, 1.8]))
    monkeypatch.setenv("MATHMOTION_DURATIONS_FILE", str(dur_file))

    from mathmotion.manim_ext import MathMotionScene
    scene = MathMotionScene.__new__(MathMotionScene)
    # Simulate setup() reading the file
    scene._mm_durations = json.loads(dur_file.read_text())
    scene._mm_index = 0

    with scene.voiceover("first segment") as t1:
        pass
    with scene.voiceover("second segment") as t2:
        pass

    assert abs(t1.duration - 4.2) < 0.001
    assert abs(t2.duration - 1.8) < 0.001


def test_voiceover_increments_index(tmp_path, monkeypatch):
    dur_file = tmp_path / "durations.json"
    dur_file.write_text(json.dumps([1.0, 2.0, 3.0]))
    monkeypatch.setenv("MATHMOTION_DURATIONS_FILE", str(dur_file))

    from mathmotion.manim_ext import MathMotionScene
    scene = MathMotionScene.__new__(MathMotionScene)
    scene._mm_durations = [1.0, 2.0, 3.0]
    scene._mm_index = 0

    durations = []
    for text in ["a", "b", "c"]:
        with scene.voiceover(text) as tracker:
            durations.append(tracker.duration)

    assert durations == [1.0, 2.0, 3.0]


def test_voiceover_falls_back_after_durations_exhausted(monkeypatch):
    monkeypatch.delenv("MATHMOTION_DURATIONS_FILE", raising=False)
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
