# Upload Script Feature Design

**Date:** 2026-03-10
**Status:** Approved

## Overview

Allow users to upload a `GeneratedScript` JSON file and run the pipeline from that script, skipping the LLM generation stage.

## API

New endpoint: `POST /api/generate-from-script`

- Accepts `multipart/form-data` with fields:
  - `file` — the `.json` file (required)
  - `quality` — optional, same as `/api/generate`
  - `tts_engine` — optional
  - `voice` — optional
  - `llm_provider` — optional (used for the repair loop)
- Validates the uploaded JSON using the existing `generate._validate()` function (schema + AST + forbidden imports/calls checks)
- Starts the pipeline in a background thread
- Returns `{"job_id": ...}`
- Status and download reuse existing `/api/status/{job_id}` and `/api/download/{job_id}` endpoints unchanged

## Pipeline

Add `script: Optional[GeneratedScript] = None` parameter to `pipeline.run()`.

When `script` is provided:
- Skip `generate.run()` entirely
- Write scene files to `job_dir/scenes/{scene_id}.py`
- Write `job_dir/narration.json`
- Continue with the normal stages: TTS → inject durations → render+repair loop → compose

No new function — single optional branch inside the existing `run()`.

## Frontend

Add a two-button mode toggle ("Generate" / "Upload Script") above the form.

**Generate mode** (default): existing UI unchanged.

**Upload Script mode:**
- Topic textarea hidden; styled `<input type="file" accept=".json">` shown in its place
- All other controls visible: quality, AI model, voice engine, voice
- On submit: read file via `FileReader`, do a quick client-side `JSON.parse` sanity check, then POST as `FormData` to `/api/generate-from-script`
- Progress tracking and video display reuse existing poll/status/download logic
