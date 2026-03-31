import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_tts_writes_voiceover_sidecar(tmp_path):
    """tts.run() writes {scene_id}_voiceover.json to scenes/ after synthesis."""
    from mathmotion.stages import tts as tts_stage
    from mathmotion.schemas.script import GeneratedScript

    script = GeneratedScript.model_validate({
        "title": "T", "topic": "t",
        "scenes": [{
            "id": "scene_1", "class_name": "S", "manim_code": "",
            "narration_segments": [
                {"id": "seg_1_a", "text": "Hello world today",
                 "actual_duration": None, "audio_path": None},
            ],
        }],
    })

    narration_path = tmp_path / "narration.json"
    narration_path.write_text(script.model_dump_json(indent=2))

    cfg = MagicMock()
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"
    cfg.tts.kokoro.speed = 1.0

    mock_engine = MagicMock()
    mock_engine.synthesise.return_value = 2.5

    with patch("mathmotion.stages.tts.subprocess.run"):
        tts_stage.run(script, tmp_path, cfg, mock_engine)

    sidecar_path = tmp_path / "scenes" / "scene_1_voiceover.json"
    assert sidecar_path.exists(), "sidecar file must be created by tts.run()"

    data = json.loads(sidecar_path.read_text())
    assert "seg_1_a" in data
    assert data["seg_1_a"]["duration"] == 2.5
    assert data["seg_1_a"]["audio_path"] is not None


def test_tts_sidecar_only_includes_synthesised_segments(tmp_path):
    """Sidecar omits segments with no audio_path (e.g. TTS failed)."""
    from mathmotion.stages import tts as tts_stage
    from mathmotion.schemas.script import GeneratedScript

    script = GeneratedScript.model_validate({
        "title": "T", "topic": "t",
        "scenes": [{
            "id": "scene_1", "class_name": "S", "manim_code": "",
            "narration_segments": [
                {"id": "seg_done", "text": "done",
                 "actual_duration": 1.5, "audio_path": "/tmp/done.mp3"},
                {"id": "seg_todo", "text": "todo word word",
                 "actual_duration": None, "audio_path": None},
            ],
        }],
    })

    narration_path = tmp_path / "narration.json"
    narration_path.write_text(script.model_dump_json(indent=2))

    cfg = MagicMock()
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"
    cfg.tts.kokoro.speed = 1.0

    mock_engine = MagicMock()
    mock_engine.synthesise.return_value = 2.0

    with patch("mathmotion.stages.tts.subprocess.run"):
        tts_stage.run(script, tmp_path, cfg, mock_engine)

    sidecar_path = tmp_path / "scenes" / "scene_1_voiceover.json"
    data = json.loads(sidecar_path.read_text())
    assert "seg_done" in data
    assert "seg_todo" in data  # synthesised during this run
