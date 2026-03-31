# Voiceover Pipeline Design

**Date:** 2026-03-31
**Status:** Approved

## Problem

Audio sync drifts progressively across the video. Root cause: `compute_cue_offsets` statically parses Manim scene code, estimating each `self.play()` without an explicit `run_time` as 1.0 s. These estimation errors accumulate across every animation step in every scene, causing increasing drift.

## Solution

Shift audio placement from **post-render** (ffmpeg mixing at compose time, requiring static timing analysis) to **mid-render** (Manim's `add_sound()` at animation time). Manim places audio at the exact current frame — guaranteed sync. No timing analysis needed.

## Architecture

| Stage | Before | After |
|---|---|---|
| **TTS** | Synthesize audio, record durations | Same + write `{scene_id}_voiceover.json` sidecar per scene |
| **Render** | Render silent video | Render video with embedded audio via `VoiceoverScene` |
| **Compose** | Build external audio track, ffmpeg mix | `ffmpeg concat` scene videos only |

**Removed entirely:** `inject_actual_durations`, `compute_cue_offsets`, `_build_audio_track`, `NarrationSegment.cue_offset`.

## Components

### `mathmotion/voiceover.py` (new)

```python
class VoiceoverTracker:
    def __init__(self, duration: float):
        self.duration = duration

class VoiceoverScene(Scene):
    def setup(self): ...          # loads _voiceover.json sidecar via manim.config.input_file
    def voiceover(self, seg_id):  # context manager — add_sound on enter, wait remainder on exit
        ...
```

**`voiceover(seg_id)` behaviour:**
- `__enter__`: records `start_t = self.renderer.time`, calls `self.add_sound(path)` if entry exists in sidecar
- yields `VoiceoverTracker(duration)`
- `__exit__`: computes `remaining = duration - (self.renderer.time - start_t)`, calls `self.wait(remaining)` if > 1 frame

**Graceful degradation:** if `seg_id` is missing from the sidecar (TTS failed), yield with no sound and no wait.

### Sidecar file

**Written by:** `mathmotion/stages/tts.py`, once per scene, after all segments for that scene are synthesized.
**Path:** `{job_dir}/scenes/{scene_id}_voiceover.json`
**Format:**
```json
{
  "seg_scene01_001": {
    "audio_path": "/absolute/path/to/seg_scene01_001.mp3",
    "duration": 3.45
  }
}
```

Absolute paths — render runs as a subprocess and may have a different working directory.

### LLM prompt (`prompts/scene_code.txt`)

Replace old rules 6–7 with:

```
6.  Import VoiceoverScene: from mathmotion.voiceover import VoiceoverScene
7.  Every scene class must inherit from VoiceoverScene, not Scene.
8.  For each narration segment, use: with self.voiceover("seg_id") as tracker:
9.  tracker.duration holds the audio duration — use it to size animations (e.g. run_time=tracker.duration).
10. Never use # WAIT comments or placeholder self.wait(1) calls.
```

Old rules 8–12 renumber to 11–15.

**Example LLM output:**
```python
from manim import Write, FadeIn, Text, MathTex
from mathmotion.voiceover import VoiceoverScene

class Scene_Foo(VoiceoverScene):
    def construct(self):
        text = Text("Hello")
        with self.voiceover("seg_scene01_001") as tracker:
            self.play(Write(text), run_time=tracker.duration)
        eq = MathTex("E = mc^2")
        with self.voiceover("seg_scene01_002") as tracker:
            self.play(FadeIn(eq))
```

### Schema changes (`mathmotion/schemas/script.py`)

Remove `cue_offset: float` from `NarrationSegment`. Keep `actual_duration` and `audio_path` — needed for sidecar generation.

### Pipeline cleanup (`mathmotion/pipeline.py`)

Remove:
- "Inject durations into scene code" block (calls `inject_actual_durations`)
- "Recalculate cue_offsets" block (calls `compute_cue_offsets`)
- Imports of both functions from `render.py`

### Render stage cleanup (`mathmotion/stages/render.py`)

Delete `inject_actual_durations` and `compute_cue_offsets` functions.

### Compose stage simplification (`mathmotion/stages/compose.py`)

Remove `_build_audio_track` and all helpers. `run()` simplifies to concat scene videos into `final.mp4` — no audio mixing (audio already embedded per-scene by Manim).

Rendering remains **sequential** — no subprocess parallelism or threading.

## Data Flow (after)

```
TTS stage
  → audio/segments/{scene_id}/{seg_id}.mp3   (audio files)
  → scenes/{scene_id}_voiceover.json          (sidecar: seg_id → path + duration)

Render stage (sequential)
  VoiceoverScene.setup() reads sidecar
  voiceover("seg_id") → add_sound() at exact frame → wait(remaining)
  → scenes/render/{scene_id}.mp4              (video + embedded audio)

Compose stage
  ffmpeg concat scene videos
  → output/final.mp4
```

## Files Changed

| File | Change |
|---|---|
| `mathmotion/voiceover.py` | **New** — `VoiceoverScene`, `VoiceoverTracker` |
| `mathmotion/schemas/script.py` | Remove `cue_offset` from `NarrationSegment` |
| `mathmotion/stages/tts.py` | Write `_voiceover.json` sidecar after each scene |
| `mathmotion/stages/render.py` | Delete `inject_actual_durations`, `compute_cue_offsets` |
| `mathmotion/stages/compose.py` | Remove audio track building; concat-only |
| `mathmotion/pipeline.py` | Remove duration injection and cue offset blocks |
| `prompts/scene_code.txt` | Replace rules 6–7 with voiceover rules |
