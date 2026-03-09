# Agentic Render Repair Loop — Design

**Date:** 2026-03-09

## Problem

The current pipeline validates generated Manim code with `ast.parse` and retries at generation time, but once code passes that check it is sent straight to `render.py`. When Manim itself fails (NameError, wrong API call, visual overlap, bad arguments), the render stage retries the same broken code up to 3 times then silently falls back to a title card. The LLM is never given the actual Manim error to fix.

## Goal

A reactive, LLM-in-the-loop repair cycle: try to render every scene, collect failures with their stderr, ask the LLM to fix each broken scene, then re-render — iterating until all scenes pass or repair attempts are exhausted.

## Architecture

`pipeline.py` orchestrates a new outer repair loop that replaces the internal render retry logic:

```
render all scenes → (successes, failures)
for attempt in range(repair_max_retries):
    if no failures: break
    for each failed scene:
        fix_scene(code, stderr) → validated fixed code → overwrite .py file
    render remaining failures → update successes/failures
fallback any still-failing scenes with existing title card
```

## Components

### New: `mathmotion/stages/repair.py`

- `fix_scene(scene_file: Path, stderr: str, provider: LLMProvider) -> str`
- Sends a focused repair prompt (see below) to the LLM
- Validates returned code with `ast.parse` + forbidden import/call checks
- Raises `ValidationError` if the fix is itself invalid (scene is carried to next round without overwriting the file)
- Returns the fixed Python source as a string

### Modified: `mathmotion/stages/render.py`

- Extract `try_render_all(scenes, scenes_dir, render_dir, quality, config) -> tuple[dict[str, Path], dict[str, str]]`
  - Returns `(successes, failures)` where `failures` maps `scene_id → stderr`
- Remove the internal 3-retry loop (replaced by the pipeline-level repair loop)
- Keep `_render()` and `_fallback()` unchanged

### Modified: `mathmotion/pipeline.py`

- Pass `provider` into the render+repair phase
- Drive the repair loop after generate + TTS + duration injection
- Call `_fallback()` for any scene that exhausts all repair attempts

### Modified: `config.yaml` + `mathmotion/utils/config.py`

- Add `llm.repair_max_retries` (default: `3`)

## Repair Prompt

```
You are fixing a broken Manim scene. The scene failed to render with this error:

{stderr}

Here is the current code:

{code}

Fix the code. Rules:
- Preserve the exact class name
- Preserve all # WAIT:{seg_id} comments and the self.wait() lines that follow them
- Only import what you use (no `from manim import *`)
- No forbidden imports: os, sys, subprocess, socket, urllib, requests, httpx
- No forbidden calls: open(), exec(), eval(), __import__()
- Return only the raw Python code, no markdown fences, no explanation
```

## Retry Semantics

`repair_max_retries` is a pipeline-level counter but is effectively per-scene since all failures are carried forward each round. With `repair_max_retries = 3`, every failing scene gets up to 3 LLM repair attempts.

## Failure Modes

| Failure | Handling |
|---|---|
| LLM returns invalid Python | Log, keep original file, carry scene to next repair round |
| LLM fix passes validation but still fails to render | Carry to next repair round with new stderr |
| Scene exhausts all repair attempts | Existing `_fallback()` title card |
| LLM API error during repair | Log, carry scene to next round unchanged |
