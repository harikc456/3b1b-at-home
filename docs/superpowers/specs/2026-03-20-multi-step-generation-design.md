# Multi-Step Video Generation Pipeline — Design Spec

**Date:** 2026-03-20
**Branch:** feat/multi-step-generation
**Status:** Approved

---

## Overview

Replace the current single-shot LLM generation stage (`mathmotion/stages/generate.py`) with a three-step pipeline that progressively builds up from a topic outline → per-scene scripts with animation descriptions → per-scene Manim code. Each step's output is persisted to disk before proceeding, enabling visibility, debuggability, and future resumability.

---

## Motivation

The existing pipeline generates everything in one LLM call: Manim code, narration text, scene structure, and timing — all at once. This produces outputs where:

- The LLM cannot reason about pedagogical flow before writing code
- Animation descriptions are implicit, leaving too much to chance in code generation
- A failure at any point requires re-running the entire expensive generation

The three-step approach separates concerns so each LLM call has a focused, well-scoped task.

---

## Pipeline Steps

### Step 1 — Topic Outline (`outline.py`)

**Input:** topic string, level (e.g. "undergraduate")
**LLM task:** Decide the best pedagogical structure — which concepts to cover, in what order, and why each scene exists. The LLM decides the number of scenes and their ordering autonomously based on the topic's complexity.
**Output:** `TopicOutline` (persisted as `jobs/<job_id>/outline.json`)
**Domain hints:** The `_detect_domains()` logic and `prompts/domain_hints/` files move from `generate.py` into `outline.py`. Domain hints are injected into the Step 1 prompt only.
**`response_schema`:** Pass `TopicOutline.model_json_schema()` to `provider.complete()`.

### Step 2 — Scene Scripts (`scene_script.py`)

**Input:** `TopicOutline`
**LLM task:** For each scene, write the full spoken narration dialogue AND a fully prescriptive animation description (exact object positions using Manim coordinate expressions, hex or named Manim colors, ordered steps for when each object appears, moves, highlights, dims, or disappears).
**Output:** `AllSceneScripts` (persisted as `jobs/<job_id>/scene_scripts.json`)
All scene scripts are generated in a single LLM call.
**`response_schema`:** Pass `AllSceneScripts.model_json_schema()` to `provider.complete()`.

### Step 3 — Scene Code (`scene_code.py`)

**Input:** `AllSceneScripts` + `TopicOutline`
**LLM task:** For each scene, generate a complete, self-contained Manim Python file. The LLM receives the scene's own script description plus the full outline so it knows where the scene fits in the video.
**Output:** `GeneratedScript` (existing schema, unchanged) — persisted as `narration.json` + individual scene `.py` files.
One LLM call is made per scene, sequentially. The `GeneratedScript` wrapper is assembled by `scene_code.run()` after all scenes succeed, using `title` and `topic` from `TopicOutline`.
**`response_schema`:** Pass `Scene.model_json_schema()` to `provider.complete()` for each per-scene call.

---

## Schema Design

### New schemas (added to `mathmotion/schemas/script.py`)

```python
class SceneOutlineItem(BaseModel):
    id: str           # e.g. "scene_1"
    title: str        # e.g. "Introduction to Geodesics"
    purpose: str      # what concept this scene teaches
    order: int

class TopicOutline(BaseModel):
    title: str
    topic: str
    level: str        # propagated from pipeline input so downstream stages can access it
    scenes: list[SceneOutlineItem]

class AnimationObject(BaseModel):
    id: str                  # e.g. "circle_1"
    type: str                # e.g. "Circle", "Text", "Arrow", "Axes"
    color: str               # hex e.g. "#FF6B6B" or named Manim color e.g. "BLUE"
    initial_position: str    # Manim coordinate expression e.g. "CENTER", "UP*2 + LEFT*3"

class AnimationStep(BaseModel):
    action: str       # "FadeIn", "MoveTo", "Highlight", "FadeOut", "Dim", "Write"
    target: str       # object id from AnimationObject.id
    timing: str       # e.g. "after_narration_segment_2", "simultaneously_with_step_3"
    parameters: dict[str, str | float | int]  # e.g. {"position": "RIGHT*2"}, {"color": "YELLOW"}

class AnimationDescription(BaseModel):
    objects: list[AnimationObject]
    sequence: list[AnimationStep]
    notes: str        # extra instructions for the code generator

class SceneScript(BaseModel):
    id: str
    title: str
    narration: str                       # full spoken dialogue for the scene
    animation_description: AnimationDescription

class AllSceneScripts(BaseModel):
    title: str
    topic: str
    scenes: list[SceneScript]
```

### Unchanged schemas

`NarrationSegment`, `Scene`, `GeneratedScript` — no changes. These remain the contract with the rest of the pipeline (TTS, render, compose).

---

## File Layout Changes

### New files

| Path | Purpose |
|------|---------|
| `mathmotion/stages/outline.py` | Step 1 stage: generate and persist `TopicOutline` |
| `mathmotion/stages/scene_script.py` | Step 2 stage: generate and persist `AllSceneScripts` |
| `mathmotion/stages/scene_code.py` | Step 3 stage: generate Manim code per scene, assemble `GeneratedScript` |
| `prompts/outline.txt` | System prompt for Step 1 |
| `prompts/scene_script.txt` | System prompt for Step 2 |
| `prompts/scene_code.txt` | System prompt for Step 3 |

### Removed files

| Path | Reason |
|------|--------|
| `mathmotion/stages/generate.py` | Fully superseded by the three new stage files |
| `prompts/system_prompt.txt` | Replaced by three focused prompt files |

### Files requiring updates

| Path | Change required |
|------|----------------|
| `mathmotion/pipeline.py` | Replace `generate` import and call with three new stage calls (see Pipeline section) |
| `mathmotion/stages/__init__.py` | Remove `generate` from any explicit imports/exports; add `outline`, `scene_script`, `scene_code` if listed |
| `api/routes.py` line 86 | Change `from mathmotion.stages.generate import validate_script` → `from mathmotion.utils.validation import validate_script` |

### Files kept unchanged

`prompts/domain_hints/` — retained and used by `outline.py` (Step 1).

---

## Prompt Responsibilities

**`prompts/outline.txt`**
- Receives: `{topic}`, `{level}`, `{domain_hints}`, `{schema_json}` (TopicOutline schema)
- Instructs the LLM to think pedagogically: what concepts must be covered, in what order, why each scene exists
- Output: ordered list of scenes with id, title, purpose, order
- No code, no narration — pure structure

**`prompts/scene_script.txt`**
- Receives: `{outline_json}` (full TopicOutline), `{schema_json}` (AllSceneScripts schema)
- Instructs the LLM to write narration dialogue (spoken language, no LaTeX/Unicode math symbols — "x squared" not "x²")
- Instructs the LLM to write fully prescriptive animation descriptions:
  - All objects with type, color (hex or named Manim color), and initial position (Manim coordinate expression)
  - All steps with action, target object id, timing relative to narration segments or other steps, and typed parameters
- Output: `AllSceneScripts`

**`prompts/scene_code.txt`**
- Receives: `{outline_json}` (full TopicOutline), `{scene_script_json}` (one SceneScript), `{schema_json}` (Scene schema)
- Instructs the LLM to produce a complete, self-contained Manim Python file implementing the animation description
- Carries forward all existing code constraints:
  - Forbidden imports: `os`, `sys`, `subprocess`, `socket`, `urllib`, `requests`, `httpx`
  - Forbidden calls: `open()`, `exec()`, `eval()`, `__import__()`
  - Unique class names starting with `Scene_`
  - `# WAIT:{segment_id}` comment before every narration-aligned `self.wait()`
  - `self.wait()` always has value 1 (replaced with real TTS durations before rendering)
  - Never use `from manim import *`, import only necessary modules
- Output: `Scene` (manim_code + narration_segments)

---

## Validation

### Per-stage validation

| Stage | Validation |
|-------|-----------|
| `outline.py` | Schema validation against `TopicOutline`; non-empty scenes list |
| `scene_script.py` | Schema validation against `AllSceneScripts`; non-empty narration per scene |
| `scene_code.py` | Schema validation against `Scene`; `ast.parse` syntax check; `check_forbidden_imports`; `check_forbidden_calls`; class name present in code; no empty narration segment text |

### `validate_script` relocation

`validate_script` (currently in `generate.py`) moves to `mathmotion/utils/validation.py` — it belongs alongside the other validation helpers (`check_forbidden_imports`, `check_forbidden_calls`). It is called by `scene_code.py` internally and remains importable for `api/routes.py` from its new location.

---

## Error Handling and Retry

Each stage uses the same retry loop pattern as the existing `generate.py`:

- Up to `config.llm.max_retries` attempts
- On failure, the error message is appended to the next attempt's user prompt
- A stage raises `LLMError` if all retries are exhausted

### Step 3 per-scene failure contract

`scene_code.run()` iterates scenes sequentially. If a scene exhausts all retries, it is recorded as a failure but the loop continues to the next scene. After all scenes are processed:

- If any scenes failed, `scene_code.run()` raises `LLMError` with a summary of which scenes failed.
- Successfully generated scenes are NOT written to disk until all scenes succeed — this covers both the individual `scenes/scene_N.py` files and `narration.json`. Nothing is written until the full `GeneratedScript` can be assembled cleanly.
- This mirrors the existing pattern: the render-repair loop handles render failures with a fallback, but code generation failure is terminal (the video cannot be rendered without code).

---

## Pipeline Orchestrator Changes (`mathmotion/pipeline.py`)

### Import update

```python
# Remove:
from mathmotion.stages import generate, render, repair, tts, compose

# Add:
from mathmotion.stages import outline as outline_stage
from mathmotion.stages import scene_script as scene_script_stage
from mathmotion.stages import scene_code as scene_code_stage
from mathmotion.stages import render, repair, tts, compose
```

### Generation call update

```python
# Before
script = generate.run(topic, job_dir, config, provider, level=level)

# After
outline_result = outline_stage.run(topic, job_dir, config, provider, level=level)
scripts_result = scene_script_stage.run(outline_result, job_dir, config, provider)
script = scene_code_stage.run(scripts_result, outline_result, job_dir, config, provider)
```

### `script=` bypass path

The existing `script=` parameter in `pipeline.run()` allows callers to skip generation entirely (used by the `/generate-from-script` API endpoint). This path is unchanged: when `script` is provided, all three generation stages are skipped. The caller is responsible for validating the script using `mathmotion.utils.validation.validate_script` before passing it in.

### Complete updated progress table

All six `progress()` call sites in `pipeline.py` are updated. This table replaces the existing values entirely:

| % | Step | Label (bypass path, `script=` provided) |
|---|------|------------------------------------------|
| 10 | "Generating outline" | "Preparing script" |
| 20 | "Writing scene scripts" | *(skipped)* |
| 35 | "Generating scene code" | *(skipped)* |
| 45 | "Synthesising audio" | "Synthesising audio" |
| 60 | "Injecting durations into scene code" | "Injecting durations into scene code" |
| 70 | "Rendering animation" | "Rendering animation" |
| 88 | "Composing final video" | "Composing final video" |
| 100 | "Done" | "Done" |

When `script=` is provided the pipeline emits `"Preparing script"` at 10% then jumps directly to 45% for TTS — the 20% and 35% calls are not emitted. This preserves the existing behavior (generation skipped entirely) while fitting the new progress scale.

---

## Disk Artifacts Per Job

After the three steps complete, `jobs/<job_id>/` contains:

```
outline.json          ← Step 1 output (TopicOutline)
scene_scripts.json    ← Step 2 output (AllSceneScripts)
narration.json        ← Step 3 output (GeneratedScript, same as today)
scenes/
  scene_1.py
  scene_2.py
  ...
```

---

## Out of Scope

- Parallelising Step 3 code generation across scenes (sequential for now)
- Skip/resume logic (re-run from a specific step without re-running earlier steps)
- UI changes to expose intermediate outputs

These are natural follow-ons once the three-step pipeline is working.
