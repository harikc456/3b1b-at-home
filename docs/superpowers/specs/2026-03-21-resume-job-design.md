# Resume Interrupted Job — Design Spec

**Date:** 2026-03-21
**Status:** Approved

---

## Overview

Add the ability to resume a pipeline job that was interrupted — either by a server crash/restart or by a mid-pipeline failure. The user selects which stage to restart from; all prior stages load their saved outputs from disk rather than re-running.

---

## Background

The mathmotion pipeline runs these stages in order, each persisting output to disk:

| Stage | Output file(s) |
|---|---|
| `outline` | `{job_dir}/outline.json` |
| `scene_script` | `{job_dir}/scene_scripts.json` |
| `scene_code` | `{job_dir}/scenes/scene_N.py` (one per scene) + `{job_dir}/narration.json` (no durations) |
| `tts` | `{job_dir}/audio/segments/**`; mutates `{job_dir}/narration.json` in-place (adds `actual_duration` + `audio_path` per segment) |
| `render` | `{job_dir}/scenes/render/{scene_id}.mp4` (stem = scene ID string from `GeneratedScript`) |
| `compose` | `{job_dir}/final.mp4` (or similar) |

`narration.json` lifecycle: written by `scene_code` (no durations). Before `tts` starts, pipeline copies it to `narration.json.bak` **only if `narration.json.bak` does not already exist** — this preserves the original pre-TTS state across multiple resume attempts and is never overwritten on subsequent resumes. The backup is forensic only and is never auto-restored. `tts` then mutates `narration.json` segment-by-segment. `actual_duration` is an `Optional[float]` field on each segment (already Optional in `GeneratedScript`), so both the pre-TTS and post-TTS forms deserialize cleanly with `GeneratedScript.model_validate`.

Job status is currently in-memory only and lost on server restart.

---

## Job status enum

| Value | Meaning |
|---|---|
| `running` | pipeline is executing |
| `complete` | finished successfully |
| `failed` | raised an exception |

**Startup reset:** The server startup sequence (ASGI lifespan `startup` event, which completes before any requests are accepted) scans all job directories, resets any `status=running` to `status=failed` with `error="Server restarted while job was running"`, and deletes any stale `job_state.json.tmp` files.

---

## Design

### 1. Stage skip logic — `pipeline.py`

```python
STAGES = ["outline", "scene_script", "scene_code", "tts", "render", "compose"]
```

`pipeline.run()` gains `start_from_stage: Optional[str] = None`.

**Semantics of `start_from_stage`:** the named stage and all later stages run; all earlier stages are skipped (their outputs are loaded from disk). `None` or `"outline"` both mean: run all stages, no files required to pre-exist.

`_should_run(stage, start_from_stage)`:
- Returns `True` if `start_from_stage` is `None`
- Returns `True` if `STAGES.index(stage) >= STAGES.index(start_from_stage)`
- Returns `False` otherwise

**Pre-flight validation** runs in the **route handler**, before the background thread is spawned, so errors are returned synchronously. For each stage that will be skipped (i.e. stage index < `start_from_stage` index), validate:

| Skipped stage | Required files |
|---|---|
| `outline` | `outline.json` exists |
| `scene_script` | `scene_scripts.json` exists |
| `scene_code` | `narration.json` exists; `scenes/{scene_id}.py` exists for every scene ID in `narration.json` |
| `tts` | `narration.json` exists with `actual_duration` (not null) on every segment |
| `render` | `narration.json` exists; `scenes/render/{scene_id}.mp4` exists for every scene ID in `narration.json` |

Validation for `scene_code`, `tts`, and `render` all require loading `narration.json` to enumerate scene IDs. This is acceptable — `narration.json` is small.

`compose` **can** be passed as `start_from_stage`. Its required pre-existing files are: `narration.json` exists; `scenes/render/{scene_id}.mp4` exists for every scene ID in `narration.json`.

**Load from disk per skipped stage:**

| Stage | Load from disk |
|---|---|
| `outline` | `TopicOutline.model_validate(json.loads(outline.json))` |
| `scene_script` | `AllSceneScripts.model_validate(json.loads(scene_scripts.json))` |
| `scene_code` | `GeneratedScript.model_validate(json.loads(narration.json))` |
| `tts` | `GeneratedScript.model_validate(json.loads(narration.json))` |
| `render` | scan `scenes/render/*.mp4`; build `{stem: Path}` dict (stem = scene ID); validate all scene IDs from `GeneratedScript` are present |

`compose` always runs when reached. It receives the render results dict (either freshly rendered or loaded from disk) and validates all scene IDs from `GeneratedScript` are present before proceeding.

**`tts` idempotency on resume:** when `tts` runs (not skipped) on a job that was previously interrupted mid-TTS, it checks each segment in `narration.json` before synthesising — if `actual_duration` is already set, it skips that segment. This makes TTS resumable at segment granularity.

**`step`/`pct` during resume:** the route sets `step="Resuming…"` and `pct=0` at the start. Once the background thread begins, it updates `step`/`pct` through the normal `progress_callback` mechanism (same as a fresh run) — no changes needed there.

After success, `pipeline.run()` returns the output `Path`. The background thread stores the absolute path string in `_jobs[job_id]["output"]` and `job_state.json`.

### 2. Job state persistence

`job_state.json` written to each job directory after every state change:

```json
{
  "status": "running",
  "step": "Rendering animation",
  "pct": 70,
  "output": null,
  "error": null,
  "topic": "Fourier transforms",
  "quality": "standard",
  "level": "undergraduate",
  "voice": null,
  "tts_engine": "kokoro",
  "llm_provider": "gemini",
  "last_resumed_from_stage": null,
  "failed_at_stage": null,
  "created_at": "2026-03-21T10:30:00Z"
}
```

Fields:
- `created_at`: ISO UTC timestamp, set once at job creation, never updated.
- `last_resumed_from_stage`: the `start_from_stage` value of the most recent run (`null` = full run).
- `failed_at_stage`: set when `status` becomes `failed`; holds the name of the stage that was running when the failure occurred. Set to `null` on any successful completion or new run start.
- `output`: absolute path string to the final video file, or `null` if not yet complete.

**Concurrency:** a module-level `threading.Lock` (`_state_lock`) guards all reads and writes to both `_jobs` and `job_state.json`. All state updates go through `_update_job(job_id, **kwargs)` which acquires `_state_lock`, merges kwargs into `_jobs[job_id]`, and calls `_save_job_state`.

**`_save_job_state`:** writes `job_state.json.tmp` then `os.replace` (atomic). On write failure: log warning and swallow — the pipeline is not aborted.

**Lazy load in `GET /api/status`:** under `_state_lock`, if `job_id` absent from `_jobs`, load from disk using `_jobs.setdefault(job_id, loaded_state)`. If the loaded state has `status=running` (stale state from before startup reset had a chance to process this job), apply the `running→failed` correction at this point. Return `{"error": "Job not found"}` if disk also missing.

### 3. API changes — `api/routes.py`

**`POST /api/generate`** — no change.

**New: `POST /api/resume/{job_id}`**

Request body: `{ "start_from_stage": "tts" }`

Steps:

1. Validate `start_from_stage` in `STAGES` → HTTP 422 if not.
2. Validate `job_id` directory exists on disk → HTTP 404 if not.
3. Load `job_state.json` → HTTP 404 if missing or unreadable.
4. Run pre-flight file validation → HTTP 422 with descriptive message if files missing.
5. **Under `_state_lock`**: if `_jobs.get(job_id, {}).get("status") == "running"` → HTTP 409; else update state: `status=running`, `step="Resuming…"`, `pct=0`, `error=null`, `failed_at_stage=null`, `last_resumed_from_stage=start_from_stage`. Release lock.
6. Spawn background thread → `pipeline.run(..., start_from_stage=...)`.
7. Return `{"job_id": job_id}`.

Pre-flight validation (step 4) runs before the state is set to `running`, so no rollback is needed on validation failure. The only window where a second concurrent request could pass the 409 check is the narrow gap between step 5 and step 6 (thread spawn); if this matters, step 6 can be moved inside the lock without deadlock risk since the thread acquires the lock only when it first calls `_update_job` (not at spawn time).

**New: `GET /api/jobs`**

Scans jobs directory, reads `job_state.json` from each subdirectory. Skips directories without one. Returns up to **50** entries, sorted descending by `created_at`; ties broken by `job_id` lexicographic descending.

Response fields per entry: `job_id`, `topic`, `status`, `step`, `pct`, `error`, `created_at`, `failed_at_stage`.

**Modified: `GET /api/status/{job_id}`** — see lazy load + running→failed correction above.

**Server startup (lifespan event, before first request):**
1. Scan all job directories for `job_state.json` files with `status=running`.
2. Reset each to `status=failed`, `error="Server restarted while job was running"`.
3. Delete any `job_state.json.tmp` files in job directories.

### 4. UI changes — `static/index.html`

**Past Jobs panel** — collapsible section below status card, loaded on init via `GET /api/jobs`. Each row:
- Topic (truncated to ~60 chars)
- Status badge: green=`complete`, red=`failed`, amber=`running` (displayed as "Interrupted")
- "Resume" button for `failed` and `running` (orphaned) jobs

**Resume modal** — inline dark-card modal:
- Job ID and topic (read-only)
- Stage dropdown: options are all `STAGES` in order; default = `failed_at_stage` if set, else `last_resumed_from_stage`, else `outline`
- "Resume Job" button → `POST /api/resume/{job_id}` → poll `/status/{job_id}`

**Status card error state** — on job failure, a "Resume" button appears alongside the error message. Clicking opens the modal pre-populated with the current job.

---

## Error handling summary

| Scenario | Behaviour |
|---|---|
| `start_from_stage` invalid | HTTP 422 |
| `job_id` not found | HTTP 404 |
| `job_state.json` missing on resume | HTTP 404 |
| Resume on actively running job | HTTP 409 (under lock, no TOCTOU) |
| Required files missing (pre-flight) | HTTP 422 with descriptive message |
| `narration.json` partially mutated | `tts` skips segments with `actual_duration` already set |
| `compose` missing expected render files | Pipeline sets `status=failed` with message |
| `_save_job_state` write fails | Warning logged, swallowed |
| Stale `.tmp` on disk | Deleted on startup |
| `status=running` at startup | Reset to `failed` during startup before first request |
| Stale `running` state loaded lazily | Running→failed correction applied on lazy load |

---

## Out of scope

- Automatic stage detection (user always selects the stage explicitly)
- Authentication / multi-user isolation
- Job deletion UI
- Pagination beyond 50-job cap
