# tests/test_pipeline_resume.py
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch


STAGES = ["outline", "scene_script", "scene_code", "tts", "render", "compose"]

MINIMAL_OUTLINE = {
    "title": "Derivatives",
    "topic": "derivatives",
    "level": "undergraduate",
    "scenes": [{"id": "scene_1", "title": "Intro", "purpose": "p", "order": 1}],
}

MINIMAL_NARRATION = {
    "title": "Derivatives",
    "topic": "derivatives",
    "scenes": [{
        "id": "scene_1",
        "class_name": "Scene1",
        "manim_code": "class Scene1(Scene): pass",
        "narration_segments": [
            {"id": "seg_1_0", "text": "Hello", "cue_offset": 0.0,
             "actual_duration": 2.0, "audio_path": "/tmp/seg.mp3"},
        ],
    }],
}


def test_should_run_none_always_true():
    from mathmotion.pipeline import _should_run
    for stage in STAGES:
        assert _should_run(stage, None) is True


def test_should_run_outline_always_true():
    from mathmotion.pipeline import _should_run
    for stage in STAGES:
        assert _should_run(stage, "outline") is True


def test_should_run_tts_skips_earlier_stages():
    from mathmotion.pipeline import _should_run
    assert _should_run("outline", "tts") is False
    assert _should_run("scene_script", "tts") is False
    assert _should_run("scene_code", "tts") is False
    assert _should_run("tts", "tts") is True
    assert _should_run("render", "tts") is True
    assert _should_run("compose", "tts") is True


def test_pipeline_skips_outline_when_file_exists(tmp_path):
    """When start_from_stage='compose', all prior stages are loaded from disk."""
    from mathmotion.pipeline import run

    # Write all required pre-existing files for start_from_stage='compose'
    (tmp_path / "outline.json").write_text(json.dumps(MINIMAL_OUTLINE))
    scene_scripts = {
        "title": "Derivatives", "topic": "derivatives",
        "scenes": [{"id": "scene_1", "title": "Intro",
                    "narration": "Hello", "animation_description": {
                        "objects": [], "sequence": [], "notes": ""}}],
    }
    (tmp_path / "scene_scripts.json").write_text(json.dumps(scene_scripts))
    (tmp_path / "narration.json").write_text(json.dumps(MINIMAL_NARRATION))
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    (scenes_dir / "scene_1.py").write_text("class Scene1: pass")
    render_dir = scenes_dir / "render"
    render_dir.mkdir()
    (render_dir / "scene_1.mp4").write_bytes(b"fake")

    cfg = MagicMock()
    cfg.storage.jobs_dir = str(tmp_path.parent)
    cfg.manim.default_quality = "draft"
    cfg.llm.model = "gemini"
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"
    cfg.tts.kokoro.speed = 1.0
    cfg.tts.vibevoice.voice = "neutral"
    cfg.composition.output_preset = "ultrafast"
    cfg.composition.output_crf = 23

    # Patch compose to avoid ffmpeg
    with patch("mathmotion.stages.compose.run", return_value=tmp_path / "output" / "final.mp4") as mock_compose, \
         patch("mathmotion.stages.render.try_render_all", return_value=({}, {})), \
         patch("mathmotion.llm.factory.get_provider") as mock_provider_factory:
        run(
            topic="derivatives",
            config=cfg,
            job_id=tmp_path.name,
            start_from_stage="compose",
        )

    # Compose must be called
    mock_compose.assert_called_once()
    # Provider was constructed (needed for repair loop) but its complete method NOT called
    # (i.e., LLM generation was skipped for outline/scene_script/scene_code)
    provider_instance = mock_provider_factory.return_value
    provider_instance.complete.assert_not_called()


def test_narration_bak_created_before_tts(tmp_path):
    """narration.json.bak is created before tts if it doesn't already exist."""
    from mathmotion.pipeline import run

    # Write all files required for start_from_stage='tts'
    (tmp_path / "outline.json").write_text(json.dumps(MINIMAL_OUTLINE))
    scene_scripts = {
        "title": "Derivatives", "topic": "derivatives",
        "scenes": [{"id": "scene_1", "title": "Intro",
                    "narration": "Hello", "animation_description": {
                        "objects": [], "sequence": [], "notes": ""}}],
    }
    (tmp_path / "scene_scripts.json").write_text(json.dumps(scene_scripts))
    narration_no_durations = {
        "title": "Derivatives", "topic": "derivatives",
        "scenes": [{"id": "scene_1", "class_name": "Scene1",
                    "manim_code": "", "narration_segments": [
                        {"id": "seg_1_0", "text": "Hello", "cue_offset": 0.0}]}],
    }
    (tmp_path / "narration.json").write_text(json.dumps(narration_no_durations))
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "scene_1.py").write_text("")

    cfg = MagicMock()
    cfg.storage.jobs_dir = str(tmp_path.parent)
    cfg.manim.default_quality = "draft"
    cfg.llm.model = "gemini"
    cfg.llm.repair_max_retries = 0
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"
    cfg.tts.kokoro.speed = 1.0
    cfg.tts.vibevoice.voice = "neutral"
    cfg.composition.output_preset = "ultrafast"
    cfg.composition.output_crf = 23

    with patch("mathmotion.stages.tts.run") as mock_tts, \
         patch("mathmotion.stages.render.try_render_all", return_value=({"scene_1": tmp_path / "scenes" / "render" / "scene_1.mp4"}, {})), \
         patch("mathmotion.stages.compose.run", return_value=tmp_path / "output" / "final.mp4"), \
         patch("mathmotion.tts.factory.get_engine", return_value=MagicMock()), \
         patch("mathmotion.llm.factory.get_provider"):
        (tmp_path / "scenes" / "render").mkdir(parents=True, exist_ok=True)
        run(
            topic="derivatives",
            config=cfg,
            job_id=tmp_path.name,
            start_from_stage="tts",
        )

    assert (tmp_path / "narration.json.bak").exists()


def test_narration_bak_not_overwritten_on_resume(tmp_path):
    """If narration.json.bak already exists, it is not overwritten."""
    from mathmotion.pipeline import run

    (tmp_path / "outline.json").write_text(json.dumps(MINIMAL_OUTLINE))
    scene_scripts = {
        "title": "D", "topic": "d",
        "scenes": [{"id": "scene_1", "title": "S", "narration": "hi",
                    "animation_description": {"objects": [], "sequence": [], "notes": ""}}],
    }
    (tmp_path / "scene_scripts.json").write_text(json.dumps(scene_scripts))
    narration_no_durations = {
        "title": "D", "topic": "d",
        "scenes": [{"id": "scene_1", "class_name": "S", "manim_code": "",
                    "narration_segments": [{"id": "s", "text": "hi", "cue_offset": 0.0}]}],
    }
    (tmp_path / "narration.json").write_text(json.dumps(narration_no_durations))
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "scene_1.py").write_text("")
    original_bak = '{"original": true}'
    (tmp_path / "narration.json.bak").write_text(original_bak)

    cfg = MagicMock()
    cfg.storage.jobs_dir = str(tmp_path.parent)
    cfg.manim.default_quality = "draft"
    cfg.llm.model = "gemini"
    cfg.llm.repair_max_retries = 0
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"
    cfg.tts.kokoro.speed = 1.0

    with patch("mathmotion.stages.tts.run"), \
         patch("mathmotion.stages.render.try_render_all", return_value=({}, {})), \
         patch("mathmotion.stages.compose.run", return_value=tmp_path / "output" / "final.mp4"), \
         patch("mathmotion.tts.factory.get_engine", return_value=MagicMock()), \
         patch("mathmotion.llm.factory.get_provider"):
        (tmp_path / "scenes" / "render").mkdir(parents=True, exist_ok=True)
        run(topic="d", config=cfg, job_id=tmp_path.name, start_from_stage="tts")

    assert (tmp_path / "narration.json.bak").read_text() == original_bak


def test_narration_bak_not_created_when_tts_skipped(tmp_path):
    """When start_from_stage='compose', TTS is skipped and no .bak is created."""
    from mathmotion.pipeline import run

    (tmp_path / "outline.json").write_text(json.dumps(MINIMAL_OUTLINE))
    scene_scripts = {
        "title": "D", "topic": "d",
        "scenes": [{"id": "scene_1", "title": "S", "narration": "hi",
                    "animation_description": {"objects": [], "sequence": [], "notes": ""}}],
    }
    (tmp_path / "scene_scripts.json").write_text(json.dumps(scene_scripts))
    (tmp_path / "narration.json").write_text(json.dumps(MINIMAL_NARRATION))  # has durations
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "scene_1.py").write_text("")
    render_dir = tmp_path / "scenes" / "render"
    render_dir.mkdir()
    (render_dir / "scene_1.mp4").write_bytes(b"fake")

    cfg = MagicMock()
    cfg.storage.jobs_dir = str(tmp_path.parent)
    cfg.manim.default_quality = "draft"
    cfg.llm.model = "gemini"
    cfg.composition.output_preset = "ultrafast"
    cfg.composition.output_crf = 23

    with patch("mathmotion.stages.compose.run", return_value=tmp_path / "output" / "final.mp4"), \
         patch("mathmotion.llm.factory.get_provider"):
        run(topic="d", config=cfg, job_id=tmp_path.name, start_from_stage="compose")

    assert not (tmp_path / "narration.json.bak").exists()


def test_tts_skips_segment_with_existing_duration(tmp_path):
    """Segments that already have actual_duration are not re-synthesised."""
    from mathmotion.stages import tts as tts_stage
    from mathmotion.schemas.script import GeneratedScript

    script = GeneratedScript.model_validate({
        "title": "T", "topic": "t",
        "scenes": [{
            "id": "scene_1", "class_name": "S", "manim_code": "",
            "narration_segments": [
                {"id": "seg_done", "text": "done", "cue_offset": 0.0,
                 "actual_duration": 1.5, "audio_path": "/tmp/done.mp3"},
                {"id": "seg_todo", "text": "todo", "cue_offset": 1.5,
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

    # Only seg_todo should have been synthesised
    assert mock_engine.synthesise.call_count == 1
    call_args = mock_engine.synthesise.call_args
    assert "todo" in call_args[0][0]  # first positional arg is the text
