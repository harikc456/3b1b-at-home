import json
import pytest
from pathlib import Path


def test_tracker_exposes_duration():
    from mathmotion.voiceover import VoiceoverTracker
    tracker = VoiceoverTracker(3.5)
    assert tracker.duration == pytest.approx(3.5)


def test_load_audio_map_reads_existing_sidecar(tmp_path):
    from mathmotion.voiceover import _load_audio_map
    scene_file = tmp_path / "scene_01.py"
    sidecar = tmp_path / "scene_01_voiceover.json"
    sidecar.write_text(json.dumps({
        "seg_001": {"audio_path": "/abs/foo.mp3", "duration": 2.5}
    }))
    result = _load_audio_map(scene_file)
    assert result == {"seg_001": {"audio_path": "/abs/foo.mp3", "duration": 2.5}}


def test_load_audio_map_returns_empty_when_no_sidecar(tmp_path):
    from mathmotion.voiceover import _load_audio_map
    scene_file = tmp_path / "scene_01.py"
    assert _load_audio_map(scene_file) == {}


def test_remaining_wait_returns_positive_remainder():
    from mathmotion.voiceover import _remaining_wait
    assert _remaining_wait(duration=3.5, elapsed=1.0, frame_rate=24) == pytest.approx(2.5)


def test_remaining_wait_returns_zero_when_elapsed_exceeds_duration():
    from mathmotion.voiceover import _remaining_wait
    assert _remaining_wait(duration=2.0, elapsed=3.0, frame_rate=24) == 0.0


def test_remaining_wait_returns_zero_within_one_frame():
    # remaining = 0.03 s, one frame at 24fps = 0.0417 s → below threshold
    from mathmotion.voiceover import _remaining_wait
    assert _remaining_wait(duration=1.03, elapsed=1.0, frame_rate=24) == 0.0


def test_narration_segment_has_no_cue_offset():
    from mathmotion.schemas.script import NarrationSegment
    seg = NarrationSegment(id="seg_1", text="Hello")
    assert not hasattr(seg, "cue_offset")
