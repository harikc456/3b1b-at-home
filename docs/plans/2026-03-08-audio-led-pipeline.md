# Audio-Led Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorder the pipeline so TTS runs before render, with actual audio durations injected into Manim scene code before any video is produced.

**Architecture:** Remove all duration estimation from the schema and LLM. TTS runs first; actual durations are injected into scene files via existing `# WAIT:{seg_id}` markers; render uses the corrected files; compose becomes a pure mux. On TTS failure, fall back to a text-length estimate (`words / 2.5`).

**Tech Stack:** Python, Pydantic, ffmpeg, Manim, Kokoro/Vibevoice TTS

---

### Task 1: Strip duration fields from the schema

**Files:**
- Modify: `mathmotion/schemas/script.py`

**Step 1: Update the schema**

Replace the contents of `mathmotion/schemas/script.py` with:

```python
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
```

**Step 2: Run existing tests to see what breaks**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home
source .venv/bin/activate
pytest tests/ -v
```

Expected: some tests in `test_compose.py` reference `compute_drift` — those will be deleted later. Schema-related tests should still pass (there are none yet).

**Step 3: Commit**

```bash
git add mathmotion/schemas/script.py
git commit -m "feat: remove duration fields from schema"
```

---

### Task 2: Update the system prompt

**Files:**
- Modify: `prompts/system_prompt.txt`

**Step 1: Edit the system prompt**

Make these two changes:
- Delete rule 6 entirely: `Keep narration segments short: 1–3 sentences, 3–6 seconds each.`
- Replace rule 10 with: `self.wait() values must always be 1 — they are replaced automatically with real TTS durations before rendering.`

Renumber rules 7–10 to 6–9 after deleting old rule 6.

The final prompt should look like:

```
You are an expert mathematical animator and educator. Generate a complete Manim
animation script and narration for the following topic.

TOPIC: {topic}
LEVEL: {level}
DURATION POLICY: Do NOT target a fixed duration. Determine the appropriate length
based on the topic's complexity and the student's LEVEL. High school explanations
need more scaffolding and analogies; graduate content can be denser and faster.
Teach the concept completely — no artificial padding or premature truncation.

DOMAIN HINTS:
{domain_hints}

OUTPUT RULES:
1. Respond with a single JSON object only. No markdown fences. No explanation.
2. Conform exactly to this JSON schema:
{schema_json}
3. Each scene's manim_code must be a complete, self-contained Python file.
4. All class names must be unique and start with "Scene_".
5. Narration text must be spoken language — no LaTeX, no Unicode math symbols.
   Write "x squared plus one" not "x² + 1". Write "pi" not "π".
6. Every narration-aligned self.wait() must be preceded by: # WAIT:{{segment_id}}
7. Forbidden imports: os, sys, subprocess, socket, urllib, requests, httpx.
8. Forbidden calls: open(), exec(), eval(), __import__().
9. self.wait() values must always be 1 — they are replaced automatically with real TTS durations before rendering.
```

**Step 2: Commit**

```bash
git add prompts/system_prompt.txt
git commit -m "feat: update system prompt for audio-led pipeline"
```

---

### Task 3: Update the TTS stage to use text-length fallback

**Files:**
- Modify: `mathmotion/stages/tts.py`

**Step 1: Write a failing test for the text-length fallback**

Add to a new file `tests/test_tts_fallback.py`:

```python
def test_text_length_duration():
    text = "This is a ten word sentence that we use for testing."
    words = len(text.split())
    duration = words / 2.5
    assert abs(duration - words / 2.5) < 0.01
```

Run it:

```bash
pytest tests/test_tts_fallback.py -v
```

Expected: PASS (it's testing the formula directly).

**Step 2: Update `tts.py`**

Replace the `synth` inner function in `run()`. The only change is the fallback: replace `seg.estimated_duration` with `len(seg.text.split()) / 2.5`:

```python
def synth(scene_id: str, seg: NarrationSegment):
    logger.info(f"Synthesising {seg.id} ({len(seg.text)} chars)…")
    out = audio_dir / seg.id
    fallback_duration = len(seg.text.split()) / 2.5
    try:
        duration = engine.synthesise(seg.text, out, voice=voice, speed=speed)
        mp3 = _wav_to_mp3(out.with_suffix(".wav"))
    except (TTSError, Exception) as e:
        logger.error(f"TTS failed for {seg.id}: {e} — using silence")
        mp3 = _silence(out, fallback_duration)
        duration = fallback_duration
    return seg.id, duration, str(mp3)
```

**Step 3: Run tests**

```bash
pytest tests/ -v
```

Expected: all pass.

**Step 4: Commit**

```bash
git add mathmotion/stages/tts.py tests/test_tts_fallback.py
git commit -m "feat: use text-length fallback in TTS stage"
```

---

### Task 4: Update the render fallback to use text-length estimate

**Files:**
- Modify: `mathmotion/stages/render.py`

**Step 1: Update `_fallback()`**

Replace the `dur=` line in `_fallback()`. Currently:
```python
dur=max(3, int(scene.estimated_duration)),
```

Change to:
```python
dur=max(3, int(sum(len(seg.text.split()) / 2.5 for seg in scene.narration_segments))),
```

The full updated `_fallback` function:

```python
def _fallback(scene: Scene, scenes_dir: Path, render_dir: Path, config) -> Path:
    dur = max(3, int(sum(len(seg.text.split()) / 2.5 for seg in scene.narration_segments)))
    code = FALLBACK_TEMPLATE.format(
        sid=scene.id,
        title=scene.id.replace("_", " ").title(),
        dur=dur,
    )
    fb_scene = Scene(
        id=scene.id,
        class_name=f"Scene_Fallback_{scene.id}",
        manim_code=code,
        narration_segments=scene.narration_segments,
    )
    (scenes_dir / f"{scene.id}.py").write_text(code)
    return _render(fb_scene, scenes_dir, render_dir, "draft", config)
```

**Step 2: Run tests**

```bash
pytest tests/ -v
```

Expected: all pass.

**Step 3: Commit**

```bash
git add mathmotion/stages/render.py
git commit -m "feat: update render fallback to use text-length duration"
```

---

### Task 5: Move `inject_actual_durations` out of compose and into a shared location

The inject logic is currently in `compose.py`. It needs to be callable from the pipeline before render. Move it to `mathmotion/stages/render.py` (alongside the render logic it feeds into), and update the import in tests.

**Files:**
- Modify: `mathmotion/stages/render.py`
- Modify: `mathmotion/stages/compose.py`
- Modify: `tests/test_compose.py`

**Step 1: Move `inject_actual_durations` into `render.py`**

Add this function to `mathmotion/stages/render.py` (after the imports, before `_render`):

```python
import re

def inject_actual_durations(code: str, durations: dict[str, float]) -> str:
    """Replace self.wait() values on lines following # WAIT:{seg_id} comments."""
    lines, out, i = code.split("\n"), [], 0
    while i < len(lines):
        m = re.search(r"# WAIT:(\S+)", lines[i])
        if m and (i + 1) < len(lines) and m.group(1) in durations:
            out.append(lines[i])
            out.append(re.sub(
                r"self\.wait\(.*?\)",
                f"self.wait({durations[m.group(1)]:.3f})",
                lines[i + 1],
            ))
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)
```

Note: `re` is already imported in `render.py`? Check — if not, add `import re` at the top.

**Step 2: Update test imports**

In `tests/test_compose.py`, update the import line:

```python
from mathmotion.stages.render import inject_actual_durations
```

Remove the `compute_drift` import and delete the two drift tests (`test_drift_calculation`, `test_drift_zero_estimated`) since `compute_drift` is being removed.

The final `tests/test_compose.py` should be:

```python
from mathmotion.stages.render import inject_actual_durations


def test_inject_replaces_wait():
    code = "# WAIT:seg_001_a\nself.wait(3.8)"
    result = inject_actual_durations(code, {"seg_001_a": 4.2})
    assert "self.wait(4.200)" in result


def test_inject_ignores_unknown_segment():
    code = "# WAIT:seg_999\nself.wait(3.0)"
    result = inject_actual_durations(code, {"seg_001_a": 4.2})
    assert "self.wait(3.0)" in result


def test_inject_leaves_unrelated_lines():
    code = "x = 1\n# WAIT:seg_001_a\nself.wait(3.0)\ny = 2"
    result = inject_actual_durations(code, {"seg_001_a": 5.0})
    assert "x = 1" in result
    assert "y = 2" in result
    assert "self.wait(5.000)" in result
```

**Step 3: Run tests**

```bash
pytest tests/test_compose.py -v
```

Expected: 3 inject tests PASS.

**Step 4: Commit**

```bash
git add mathmotion/stages/render.py tests/test_compose.py
git commit -m "feat: move inject_actual_durations to render module"
```

---

### Task 6: Simplify compose to a pure mux

**Files:**
- Modify: `mathmotion/stages/compose.py`

**Step 1: Rewrite `compose.py`**

Replace the entire file with:

```python
import logging
import subprocess
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import CompositionError

logger = logging.getLogger(__name__)


def _silence(path: Path, duration: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(duration), "-codec:a", "libmp3lame", "-b:a", "192k", str(path),
    ], check=True, capture_output=True)


def _build_audio_track(script: GeneratedScript, job_dir: Path) -> Path:
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    parts, current = [], 0.0

    for scene in script.scenes:
        for seg in scene.narration_segments:
            gap = seg.cue_offset - current
            if gap > 0.05:
                sil = audio_dir / f"sil_{len(parts)}.mp3"
                _silence(sil, gap)
                parts.append(str(sil))
            if seg.audio_path:
                parts.append(seg.audio_path)
            current = seg.cue_offset + (seg.actual_duration or 0.0)

    concat = audio_dir / "concat.txt"
    concat.write_text("\n".join(f"file '{p}'" for p in parts))
    out = audio_dir / "narration.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat), "-c", "copy", str(out),
    ], check=True, capture_output=True)
    return out


def run(job_dir: Path, config) -> Path:
    import json
    script = GeneratedScript.model_validate(
        json.loads((job_dir / "narration.json").read_text())
    )

    audio = _build_audio_track(script, job_dir)

    render_dir = job_dir / "scenes" / "render"
    scene_list = job_dir / "scene_list.txt"
    scene_list.write_text("\n".join(
        f"file '{render_dir / scene.id}.mp4'" for scene in script.scenes
    ))
    assembled = job_dir / "assembled.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(scene_list), "-c", "copy", str(assembled),
    ], check=True, capture_output=True)

    out_dir = job_dir / "output"
    out_dir.mkdir(exist_ok=True)
    final = out_dir / "final.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(assembled), "-i", str(audio),
            "-c:v", "libx264", "-preset", config.composition.output_preset,
            "-crf", str(config.composition.output_crf),
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            str(final),
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise CompositionError(f"FFmpeg encode failed: {e.stderr.decode()}")

    logger.info(f"Final video: {final}")
    return final
```

**Step 2: Run tests**

```bash
pytest tests/ -v
```

Expected: all pass (compose tests now import from render, compose tests still pass).

**Step 3: Commit**

```bash
git add mathmotion/stages/compose.py
git commit -m "feat: simplify compose to pure mux"
```

---

### Task 7: Update the pipeline to run TTS → inject → render sequentially

**Files:**
- Modify: `mathmotion/pipeline.py`

**Step 1: Rewrite `pipeline.py`**

Replace the parallel executor block and update progress messages:

```python
import json
import logging
import uuid
from pathlib import Path
from typing import Callable, Optional

from mathmotion.llm.factory import get_provider
from mathmotion.schemas.script import GeneratedScript
from mathmotion.stages import generate, render, tts, compose
from mathmotion.stages.render import inject_actual_durations
from mathmotion.tts.factory import get_engine
from mathmotion.utils.config import Config

logger = logging.getLogger(__name__)


def run(
    topic: str,
    config: Config,
    quality: Optional[str] = None,
    level: str = "undergraduate",
    voice: Optional[str] = None,
    tts_engine: Optional[str] = None,
    llm_provider: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Path:
    def progress(step: str, pct: int) -> None:
        logger.info(f"[{pct}%] {step}")
        if progress_callback:
            progress_callback(step, pct)

    if quality:
        config.manim.default_quality = quality
    if llm_provider:
        config.llm.provider = llm_provider
    if tts_engine:
        config.tts.engine = tts_engine
    if voice:
        if config.tts.engine == "kokoro":
            config.tts.kokoro.voice = voice
        else:
            config.tts.vibevoice.voice = voice

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = Path(config.storage.jobs_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Job {job_id} | topic={topic!r} | provider={config.llm.provider} | "
                f"tts={config.tts.engine} | quality={config.manim.default_quality}")

    progress("Generating script", 10)
    provider = get_provider(config)
    script = generate.run(topic, job_dir, config, provider, level=level)

    progress("Synthesising audio", 30)
    engine = get_engine(config)
    tts.run(script, job_dir, config, engine)

    progress("Injecting durations into scene code", 55)
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

    progress("Rendering animation", 65)
    render.run(script, job_dir, config)

    progress("Composing final video", 85)
    final = compose.run(job_dir, config)

    progress("Done", 100)
    logger.info(f"Output: {final}")
    return final
```

**Step 2: Run tests**

```bash
pytest tests/ -v
```

Expected: all pass.

**Step 3: Commit**

```bash
git add mathmotion/pipeline.py
git commit -m "feat: reorder pipeline to TTS-first with duration injection before render"
```

---

### Task 8: Remove sync fields from config

**Files:**
- Modify: `mathmotion/utils/config.py`
- Modify: `config.yaml`
- Modify: `tests/test_config.py` (check if it references these fields)

**Step 1: Check test_config.py**

Read `tests/test_config.py` to see if `sync_strategy` or `time_stretch_threshold` appear.

**Step 2: Update `CompositionConfig` in `config.py`**

Replace:
```python
class CompositionConfig(BaseModel):
    sync_strategy: str = "time_stretch"
    time_stretch_threshold: float = 0.15
    output_crf: int = 18
    output_preset: str = "slow"
```

With:
```python
class CompositionConfig(BaseModel):
    output_crf: int = 18
    output_preset: str = "slow"
```

**Step 3: Update `config.yaml`**

In the `composition:` section, remove `sync_strategy` and `time_stretch_threshold`:

```yaml
composition:
  output_crf: 18
  output_preset: slow
```

**Step 4: Run tests**

```bash
pytest tests/ -v
```

Expected: all pass.

**Step 5: Commit**

```bash
git add mathmotion/utils/config.py config.yaml
git commit -m "feat: remove sync config fields"
```

---

### Task 9: Final check — run full test suite

**Step 1: Run all tests**

```bash
cd /home/harikrishnan-c/projects/3b1b-at-home
source .venv/bin/activate
pytest tests/ -v
```

Expected: all tests pass, no references to `estimated_duration`, `compute_drift`, `sync_strategy`, or `time_stretch_threshold` in active code.

**Step 2: Verify no leftover references**

```bash
grep -r "estimated_duration\|sync_strategy\|time_stretch\|compute_drift" mathmotion/ prompts/
```

Expected: no output.

**Step 3: Commit if any cleanup needed, then push**

```bash
git log --oneline -10
```
