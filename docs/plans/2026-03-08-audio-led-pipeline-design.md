# Audio-Led Pipeline Design

**Date:** 2026-03-08

## Problem

The LLM was responsible for estimating scene and segment durations, which were used to set `self.wait()` values in Manim code. Video was rendered before (or in parallel with) audio, so duration estimates were always guesses. Most renders fell back to title cards because the Manim code failed, and the fallback used the estimated duration. The sync stage (time_stretch / audio_led re-render) existed purely to correct this mismatch.

## Goal

Let the LLM focus entirely on script quality. Derive all durations from actual TTS output. Render video with real durations already baked in. Compose is a pure mux.

## Design

### Schema (`mathmotion/schemas/script.py`)

Remove:
- `NarrationSegment.estimated_duration`
- `Scene.estimated_duration`
- `GeneratedScript.total_estimated_duration`

Keep:
- `NarrationSegment.actual_duration: Optional[float]` — filled in by TTS stage

### Pipeline Order (`mathmotion/pipeline.py`)

```
generate → TTS → inject durations → render → compose (mux only)
```

Previously render and TTS ran in parallel. Now TTS runs first, then durations are injected into scene code, then render.

### TTS Stage (`mathmotion/stages/tts.py`)

On TTS failure for a segment, fall back to a text-length estimate:

```python
duration = len(seg.text.split()) / 2.5  # ~150 wpm
```

No longer references `seg.estimated_duration`.

### Inject Step

`compose.inject_actual_durations` already exists and rewrites `self.wait(X)` values using `# WAIT:{seg_id}` markers. This logic moves to run between TTS and render in the pipeline. It is no longer called from compose.

### Render Fallback (`mathmotion/stages/render.py`)

`_fallback()` used `scene.estimated_duration` for `self.wait()`. Replace with:

```python
dur = sum(len(seg.text.split()) / 2.5 for seg in scene.narration_segments)
dur = max(3, dur)
```

### Compose Stage (`mathmotion/stages/compose.py`)

Remove entirely:
- `sync_strategy` branching (`time_stretch`, `audio_led`)
- `compute_drift()`
- `inject_actual_durations()` (moved to pipeline)

Compose becomes: build audio track → concat video scenes → ffmpeg mux.

### Config (`config.yaml`)

Remove from `composition`:
- `sync_strategy`
- `time_stretch_threshold`

### System Prompt (`prompts/system_prompt.txt`)

- Remove rule 6 ("3–6 seconds each") — segment length is no longer duration-constrained
- Update rule 10: `self.wait()` values must always be `1` — they are replaced with real TTS durations before rendering
- Schema embedded in prompt will no longer include duration fields

## Trade-offs

- **No parallelism between TTS and render** — accepted, since correctness matters more than speed here
- **TTS failure fallback is approximate** — a text-length estimate is better than nothing and avoids requiring `estimated_duration` to remain in the schema
- **Compose simplification** — removing sync logic reduces complexity and eliminates a whole class of bugs
