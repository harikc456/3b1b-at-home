# Voiceover Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static `cue_offset` audio alignment system with `VoiceoverScene` — a Manim `Scene` subclass that embeds audio directly at render time, eliminating progressive audio drift.

**Architecture:** TTS runs first (unchanged), then writes a `{scene_id}_voiceover.json` sidecar with audio paths and durations per segment. During rendering, `VoiceoverScene.voiceover(seg_id)` calls Manim's `add_sound()` at the exact current frame, then waits for any remaining audio duration. The compose stage simplifies to a single `ffmpeg concat` — no external audio track.

**Tech Stack:** Python 3.13, Manim (Cairo renderer), FFmpeg, Pydantic v2, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `mathmotion/voiceover.py` | **Create** | `VoiceoverTracker`, pure helpers, `VoiceoverScene` |
| `tests/test_voiceover.py` | **Create** | Unit tests for tracker and pure helpers |
| `mathmotion/schemas/script.py` | **Modify** | Remove `cue_offset` from `NarrationSegment` |
| `mathmotion/stages/tts.py` | **Modify** | Write `_voiceover.json` sidecar after all segments synthesized |
| `tests/test_pipeline_resume.py` | **Modify** | Remove `cue_offset` from fixture dicts |
| `mathmotion/stages/render.py` | **Modify** | Delete `inject_actual_durations` and `compute_cue_offsets` |
| `tests/test_compose.py` | **Delete** | Tests only deleted functions |
| `mathmotion/stages/compose.py` | **Modify** | Remove audio track building; concat-only |
| `tests/test_stage_compose.py` | **Create** | Tests for simplified compose |
| `mathmotion/pipeline.py` | **Modify** | Remove inject-durations + cue-offset blocks and imports |
| `prompts/scene_code.txt` | **Modify** | Replace old rules 6–7 with voiceover rules |

---

### Task 1: VoiceoverTracker and pure helper functions

**Files:**
- Create: `mathmotion/voiceover.py`
- Create: `tests/test_voiceover.py`

- [ ] **Step 1: Write failing tests for VoiceoverTracker and helpers**

Create `tests/test_voiceover.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_voiceover.py -v
```

Expected: `ModuleNotFoundError: No module named 'mathmotion.voiceover'`

- [ ] **Step 3: Implement the pure functions (no Manim dependency)**

Create `mathmotion/voiceover.py`:

```python
import json
from contextlib import contextmanager
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_voiceover.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mathmotion/voiceover.py tests/test_voiceover.py
git commit -m "feat: add VoiceoverTracker and pure voiceover helpers"
```

---

### Task 2: VoiceoverScene class

**Files:**
- Modify: `mathmotion/voiceover.py`

- [ ] **Step 1: Replace the placeholder class with the full implementation**

Edit `mathmotion/voiceover.py` — replace the `VoiceoverScene` placeholder with:

```python
from manim import Scene, config as manim_config


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
```

The final `mathmotion/voiceover.py` should look like:

```python
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
```

- [ ] **Step 2: Confirm existing voiceover tests still pass**

```bash
uv run pytest tests/test_voiceover.py -v
```

Expected: all 6 tests PASS (the pure-function tests are unaffected by the class addition).

- [ ] **Step 3: Commit**

```bash
git add mathmotion/voiceover.py
git commit -m "feat: implement VoiceoverScene with add_sound context manager"
```

---

### Task 3: Remove `cue_offset` from schema

**Files:**
- Modify: `mathmotion/schemas/script.py`
- Modify: `tests/test_pipeline_resume.py`

- [ ] **Step 1: Write a failing test confirming cue_offset is gone**

Add to `tests/test_voiceover.py`:

```python
def test_narration_segment_has_no_cue_offset():
    from mathmotion.schemas.script import NarrationSegment
    seg = NarrationSegment(id="seg_1", text="Hello")
    assert not hasattr(seg, "cue_offset")
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_voiceover.py::test_narration_segment_has_no_cue_offset -v
```

Expected: FAIL — `NarrationSegment` currently has `cue_offset`.

- [ ] **Step 3: Remove `cue_offset` from NarrationSegment**

In `mathmotion/schemas/script.py`, change:

```python
class NarrationSegment(BaseModel):
    id: str
    text: str
    cue_offset: float
    actual_duration: Optional[float] = None
    audio_path: Optional[str] = None
```

to:

```python
class NarrationSegment(BaseModel):
    id: str
    text: str
    actual_duration: Optional[float] = None
    audio_path: Optional[str] = None
```

- [ ] **Step 4: Clean up cue_offset from test fixtures in test_pipeline_resume.py**

In `tests/test_pipeline_resume.py`, update `MINIMAL_NARRATION`:

```python
MINIMAL_NARRATION = {
    "title": "Derivatives",
    "topic": "derivatives",
    "scenes": [{
        "id": "scene_1",
        "class_name": "Scene1",
        "manim_code": "class Scene1(Scene): pass",
        "narration_segments": [
            {"id": "seg_1_0", "text": "Hello",
             "actual_duration": 2.0, "audio_path": "/tmp/seg.mp3"},
        ],
    }],
}
```

Also update the inline fixture in `test_narration_bak_created_before_tts`:

```python
narration_no_durations = {
    "title": "Derivatives", "topic": "derivatives",
    "scenes": [{"id": "scene_1", "class_name": "Scene1",
                "manim_code": "", "narration_segments": [
                    {"id": "seg_1_0", "text": "Hello"}]}],
}
```

Also update the inline fixture in `test_narration_bak_not_overwritten_on_resume`:

```python
narration_no_durations = {
    "title": "D", "topic": "d",
    "scenes": [{"id": "scene_1", "class_name": "S", "manim_code": "",
                "narration_segments": [{"id": "s", "text": "hi"}]}],
}
```

Also update `test_tts_skips_segment_with_existing_duration` fixture — the `GeneratedScript.model_validate` call:

```python
script = GeneratedScript.model_validate({
    "title": "T", "topic": "t",
    "scenes": [{
        "id": "scene_1", "class_name": "S", "manim_code": "",
        "narration_segments": [
            {"id": "seg_done", "text": "done",
             "actual_duration": 1.5, "audio_path": "/tmp/done.mp3"},
            {"id": "seg_todo", "text": "todo",
             "actual_duration": None, "audio_path": None},
        ],
    }],
})
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all previously-passing tests still pass, new schema test passes.

- [ ] **Step 6: Commit**

```bash
git add mathmotion/schemas/script.py tests/test_voiceover.py tests/test_pipeline_resume.py
git commit -m "feat: remove cue_offset from NarrationSegment schema"
```

---

### Task 4: TTS stage — write voiceover sidecar

**Files:**
- Modify: `mathmotion/stages/tts.py`
- Create: `tests/test_stage_tts_sidecar.py`

- [ ] **Step 1: Write a failing test for sidecar writing**

Create `tests/test_stage_tts_sidecar.py`:

```python
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
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
uv run pytest tests/test_stage_tts_sidecar.py -v
```

Expected: FAIL — `tts.run()` does not currently write a sidecar file.

- [ ] **Step 3: Add sidecar writing to tts.run()**

In `mathmotion/stages/tts.py`, add a sidecar-writing block at the **end** of `run()`, after the synthesis loop. The full updated `run()` function:

```python
def run(script: GeneratedScript, job_dir: Path, config, engine: TTSEngine) -> None:
    audio_dir = job_dir / "audio" / "segments"
    audio_dir.mkdir(parents=True, exist_ok=True)

    match config.tts.engine:
        case "kokoro":    tts_cfg = config.tts.kokoro
        case "vibevoice": tts_cfg = config.tts.vibevoice
        case n:           raise ValueError(f"Unknown TTS engine: {n!r}")
    voice = tts_cfg.voice
    speed = tts_cfg.speed

    segments = [
        (scene.id, seg)
        for scene in script.scenes
        for seg in scene.narration_segments
    ]
    total_segs = len(segments)
    logger.debug(f"TTS run: engine={config.tts.engine!r}, scenes={len(script.scenes)}, total_segments={total_segs}, voice={voice!r}, speed={speed}")
    logger.info(f"Synthesising {total_segs} audio segment(s) with engine={config.tts.engine!r}")

    narration_path = job_dir / "narration.json"
    if not narration_path.exists():
        narration_path.write_text(script.model_dump_json(indent=2))

    def synth(scene_id: str, seg: NarrationSegment):
        logger.info(f"Synthesising {seg.id} ({len(seg.text)} chars)…")
        scene_audio_dir = audio_dir / scene_id
        scene_audio_dir.mkdir(parents=True, exist_ok=True)
        out = scene_audio_dir / seg.id
        fallback_duration = len(seg.text.split()) / 2.5
        try:
            duration = engine.synthesise(seg.text, out, voice=voice, speed=speed)
            mp3 = _wav_to_mp3(out.with_suffix(".wav"))
        except (TTSError, Exception) as e:
            logger.error(f"TTS failed for {seg.id}: {e} — using silence", exc_info=True)
            mp3 = _silence(out, fallback_duration)
            duration = fallback_duration
        return scene_id, seg.id, duration, str(mp3)

    done = 0
    for sid, seg in segments:
        if seg.actual_duration is not None:
            logger.info(f"Skipping {seg.id} — already synthesised ({seg.actual_duration:.2f}s)")
            continue
        scene_id, seg_id, duration, mp3_path = synth(sid, seg)
        done += 1
        data = json.loads(narration_path.read_text())
        for scene in data["scenes"]:
            if scene["id"] != scene_id:
                continue
            for s in scene["narration_segments"]:
                if s["id"] == seg_id:
                    s["actual_duration"] = duration
                    s["audio_path"] = mp3_path
        narration_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Synthesised {seg_id} ({duration:.2f}s) — {done} new segments done")

    # Write per-scene voiceover sidecars for VoiceoverScene to consume at render time.
    scenes_dir = job_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(narration_path.read_text())
    for scene_data in data["scenes"]:
        sidecar = {
            seg["id"]: {"audio_path": seg["audio_path"], "duration": seg["actual_duration"]}
            for seg in scene_data["narration_segments"]
            if seg.get("audio_path") and seg.get("actual_duration") is not None
        }
        sidecar_path = scenes_dir / f"{scene_data['id']}_voiceover.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
        logger.info(f"Wrote voiceover sidecar: {sidecar_path.name} ({len(sidecar)} segment(s))")
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass including the two new sidecar tests.

- [ ] **Step 5: Commit**

```bash
git add mathmotion/stages/tts.py tests/test_stage_tts_sidecar.py
git commit -m "feat: write voiceover sidecar JSON after TTS synthesis"
```

---

### Task 5: Delete dead render functions and their tests

**Files:**
- Modify: `mathmotion/stages/render.py`
- Delete: `tests/test_compose.py`

- [ ] **Step 1: Delete `tests/test_compose.py`**

```bash
git rm tests/test_compose.py
```

- [ ] **Step 2: Remove `inject_actual_durations` and `compute_cue_offsets` from render.py**

In `mathmotion/stages/render.py`, delete the two functions (lines 27–104). Also remove `import re` from the imports — it was only used by those two functions. The file should begin with imports and `QUALITY_FLAGS`, then go straight to `_render()`. The new start of the file after the deletions:

```python
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript, Scene
from mathmotion.utils.errors import RenderError

logger = logging.getLogger(__name__)

QUALITY_FLAGS = {
    "draft":    ("-ql", "854,480",   "24"),
    "standard": ("-qm", "1280,720",  "30"),
    "high":     ("-qh", "1920,1080", "60"),
}

FALLBACK_TEMPLATE = '''from manim import *
class Scene_Fallback_{sid}(Scene):
    def construct(self):
        self.play(Write(Text("{title}", font_size=48)))
        self.wait({dur})
'''


def _render(scene: Scene, scenes_dir: Path, render_dir: Path, quality: str, config) -> Path:
    ...
```

(Keep `_render`, `_fallback`, `try_render_all`, and `run` unchanged.)

- [ ] **Step 3: Confirm tests pass without the deleted functions**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass (no test references the deleted functions any more).

- [ ] **Step 4: Commit**

```bash
git add mathmotion/stages/render.py tests/test_compose.py
git commit -m "feat: remove inject_actual_durations and compute_cue_offsets"
```

---

### Task 6: Simplify compose stage to concat-only

**Files:**
- Modify: `mathmotion/stages/compose.py`
- Create: `tests/test_stage_compose.py`

- [ ] **Step 1: Write a failing test for simplified compose**

Create `tests/test_stage_compose.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch, call


def _make_narration(tmp_path):
    """Write a minimal narration.json with two scenes and return the script dict."""
    data = {
        "title": "T", "topic": "t",
        "scenes": [
            {"id": "scene_1", "class_name": "S1", "manim_code": "",
             "narration_segments": [
                 {"id": "seg_1", "text": "hello", "actual_duration": 2.0,
                  "audio_path": "/tmp/seg1.mp3"}
             ]},
            {"id": "scene_2", "class_name": "S2", "manim_code": "",
             "narration_segments": [
                 {"id": "seg_2", "text": "world", "actual_duration": 1.5,
                  "audio_path": "/tmp/seg2.mp3"}
             ]},
        ],
    }
    (tmp_path / "narration.json").write_text(json.dumps(data))
    render_dir = tmp_path / "scenes" / "render"
    render_dir.mkdir(parents=True)
    (render_dir / "scene_1.mp4").write_bytes(b"fake1")
    (render_dir / "scene_2.mp4").write_bytes(b"fake2")
    return data


def test_compose_calls_ffmpeg_concat(tmp_path):
    """compose.run() calls ffmpeg concat with the two scene files."""
    from mathmotion.stages import compose
    from unittest.mock import MagicMock

    _make_narration(tmp_path)
    cfg = MagicMock()

    with patch("mathmotion.stages.compose.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        compose.run(tmp_path, cfg)

    assert mock_run.call_count == 1
    args = mock_run.call_args[0][0]
    assert "ffmpeg" in args
    assert "-f" in args
    assert "concat" in args


def test_compose_writes_scene_list(tmp_path):
    """compose.run() writes scene_list.txt listing both scene mp4s."""
    from mathmotion.stages import compose
    from unittest.mock import MagicMock

    _make_narration(tmp_path)
    cfg = MagicMock()

    with patch("mathmotion.stages.compose.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        compose.run(tmp_path, cfg)

    scene_list = tmp_path / "scene_list.txt"
    assert scene_list.exists()
    content = scene_list.read_text()
    assert "scene_1.mp4" in content
    assert "scene_2.mp4" in content


def test_compose_returns_final_path(tmp_path):
    """compose.run() returns output/final.mp4."""
    from mathmotion.stages import compose
    from unittest.mock import MagicMock

    _make_narration(tmp_path)
    cfg = MagicMock()

    with patch("mathmotion.stages.compose.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = compose.run(tmp_path, cfg)

    assert result == tmp_path / "output" / "final.mp4"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
uv run pytest tests/test_stage_compose.py -v
```

Expected: FAIL — current `compose.run()` calls ffmpeg multiple times (audio track + encode) and imports `measure_duration`.

- [ ] **Step 3: Replace compose.py with simplified implementation**

Replace the entire contents of `mathmotion/stages/compose.py`:

```python
import json
import logging
import subprocess
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import CompositionError

logger = logging.getLogger(__name__)


def run(job_dir: Path, config) -> Path:
    script = GeneratedScript.model_validate(
        json.loads((job_dir / "narration.json").read_text())
    )

    render_dir = job_dir / "scenes" / "render"
    scene_list = job_dir / "scene_list.txt"
    scene_list.write_text("\n".join(
        f"file '{(render_dir / scene.id).with_suffix('.mp4').resolve()}'"
        for scene in script.scenes
    ))

    out_dir = job_dir / "output"
    out_dir.mkdir(exist_ok=True)
    final = out_dir / "final.mp4"

    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(scene_list), "-c", "copy", str(final),
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise CompositionError(f"FFmpeg concat failed: {e.stderr.decode()}")

    logger.info(f"Final video: {final}")
    return final
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass including the three new compose tests.

- [ ] **Step 5: Commit**

```bash
git add mathmotion/stages/compose.py tests/test_stage_compose.py
git commit -m "feat: simplify compose stage to ffmpeg concat-only"
```

---

### Task 7: Remove dead blocks from pipeline

**Files:**
- Modify: `mathmotion/pipeline.py`

- [ ] **Step 1: Remove the import of deleted functions**

In `mathmotion/pipeline.py`, remove this line:

```python
from mathmotion.stages.render import inject_actual_durations, compute_cue_offsets
```

- [ ] **Step 2: Remove the inject-durations block (lines ~173–188)**

Delete this entire block from `pipeline.py`:

```python
    # ── Inject durations ──────────────────────────────────────────────────────────
    progress("Injecting durations into scene code", 60)
    script = GeneratedScript.model_validate(
        json.loads((job_dir / "narration.json").read_text())
    )
    durations = {
        seg.id: seg.actual_duration
        for scene in script.scenes
        for seg in scene.narration_segments
        if seg.actual_duration is not None
    }
    scenes_dir = job_dir / "scenes"
    for scene in script.scenes:
        scene_file = scenes_dir / f"{scene.id}.py"
        if scene_file.exists():
            scene_file.write_text(inject_actual_durations(scene_file.read_text(), durations))
```

- [ ] **Step 3: Remove the cue-offset recalculation block (lines ~190–208)**

Delete this entire block from `pipeline.py`:

```python
    # ── Recalculate cue_offsets from actual Manim animation timings ───────────
    # The LLM-supplied cue_offset values are estimates and may not match the
    # actual elapsed time in the rendered video when each narration wait fires.
    # Re-derive them by parsing the (now duration-injected) scene code so the
    # audio track built in the compose stage stays in sync with the video.
    cue_offsets_changed = False
    for scene in script.scenes:
        scene_file = scenes_dir / f"{scene.id}.py"
        if not scene_file.exists():
            continue
        seg_ids = {seg.id for seg in scene.narration_segments}
        computed = compute_cue_offsets(scene_file.read_text(), seg_ids)
        for seg in scene.narration_segments:
            if seg.id in computed and abs(computed[seg.id] - seg.cue_offset) > 0.05:
                seg.cue_offset = round(computed[seg.id], 3)
                cue_offsets_changed = True
    if cue_offsets_changed:
        (job_dir / "narration.json").write_text(script.model_dump_json(indent=2))
        logger.info("Recalculated cue_offsets saved to narration.json")
```

After both deletions, the section between TTS and Render in `pipeline.py` should read:

```python
    # ── Render ────────────────────────────────────────────────────────────────
    if _should_run("render", start_from_stage):
        progress("Rendering animation", 70)
        _run_render_repair_loop(script, job_dir, config, provider)
    else:
        progress("Skipping render (using existing files)", 70)

    # ── Compose ───────────────────────────────────────────────────────────────
    progress("Composing final video", 88)
    final = compose.run(job_dir, config)
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add mathmotion/pipeline.py
git commit -m "feat: remove inject_durations and cue_offset recalculation from pipeline"
```

---

### Task 8: Update LLM prompt for voiceover pattern

**Files:**
- Modify: `prompts/scene_code.txt`

- [ ] **Step 1: Replace rules 6–7 and renumber the rest**

Replace the entire contents of `prompts/scene_code.txt`:

```
You are an expert Manim animation programmer. Implement a complete Manim scene from a detailed script.

VIDEO OUTLINE (context — this scene's position in the full video):
{outline_json}

SCENE SCRIPT TO IMPLEMENT:
{scene_script_json}

RULES:
1.  Respond with a single JSON object only. No markdown fences. No explanation.
2.  Conform exactly to this JSON schema:
{schema_json}
3.  manim_code must be a complete, self-contained Python file with all imports.
4.  The class name must start with "Scene_" and exactly match the class_name field.
5.  Split the narration into narration_segments at natural pause points.
6.  Import VoiceoverScene at the top of manim_code: from mathmotion.voiceover import VoiceoverScene
7.  Every scene class must inherit from VoiceoverScene, not Scene.
8.  For each narration segment use: with self.voiceover("seg_id") as tracker:
9.  tracker.duration holds the audio duration in seconds — use it to size animations (e.g. run_time=tracker.duration).
10. Never use # WAIT comments or placeholder self.wait(1) calls for narration timing.
11. Narration text must be spoken language — no LaTeX, no Unicode math symbols.
12. Forbidden imports: os, sys, subprocess, socket, urllib, requests, httpx.
13. Forbidden calls: open(), exec(), eval(), __import__().
14. Never use `from manim import *` — import only the specific classes you use.
15. Implement the animation exactly as described in animation_description.
```

- [ ] **Step 2: Verify the prompt file looks correct**

```bash
cat prompts/scene_code.txt
```

Confirm 15 rules are present, rules 6–10 reference VoiceoverScene and the voiceover context manager, no `# WAIT` references remain.

- [ ] **Step 3: Run all tests one final time**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add prompts/scene_code.txt
git commit -m "feat: update scene_code prompt to use VoiceoverScene pattern"
```
