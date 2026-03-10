# Upload Script Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow users to upload a `GeneratedScript` JSON file and run the pipeline from it, skipping LLM generation.

**Architecture:** Add a `script` param to `pipeline.run()` that, when provided, writes scene files and narration.json directly (skipping `generate.run()`). Add a `POST /api/generate-from-script` multipart endpoint that validates the uploaded JSON and kicks off the pipeline. Add a mode toggle in the frontend.

**Tech Stack:** FastAPI (UploadFile, Form), Pydantic v2, vanilla JS (FileReader, FormData)

---

### Task 1: Add `script` param to `pipeline.run()`

**Files:**
- Modify: `mathmotion/pipeline.py:71-137`
- Test: `tests/test_pipeline_upload.py`

**Step 1: Write the failing test**

```python
# tests/test_pipeline_upload.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from mathmotion.schemas.script import GeneratedScript, Scene, NarrationSegment


MINIMAL_SCENE_CODE = """from manim import *

class SceneA(Scene):
    def construct(self):
        self.wait(1)
"""

SAMPLE_SCRIPT = GeneratedScript(
    title="Test",
    topic="derivatives",
    scenes=[
        Scene(
            id="scene_a",
            class_name="SceneA",
            manim_code=MINIMAL_SCENE_CODE,
            narration_segments=[
                NarrationSegment(id="seg_1", text="Hello world", cue_offset=0.0)
            ],
        )
    ],
)


def test_pipeline_skips_generate_when_script_provided(tmp_path):
    """When script= is passed, generate.run must NOT be called."""
    from mathmotion.utils.config import Config
    cfg = MagicMock(spec=Config)
    cfg.manim.default_quality = "draft"
    cfg.storage.jobs_dir = str(tmp_path)
    cfg.llm.repair_max_retries = 0
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"

    with (
        patch("mathmotion.pipeline.get_provider") as mock_provider,
        patch("mathmotion.pipeline.get_engine"),
        patch("mathmotion.pipeline.tts.run"),
        patch("mathmotion.pipeline.inject_actual_durations", side_effect=lambda text, _: text),
        patch("mathmotion.pipeline._run_render_repair_loop", return_value={}),
        patch("mathmotion.pipeline.compose.run", return_value=tmp_path / "out.mp4"),
        patch("mathmotion.pipeline.generate.run") as mock_generate,
    ):
        # Make narration.json available for the inject_actual_durations step
        import uuid
        job_id = None

        original_run = None
        from mathmotion import pipeline as _pl
        _original = _pl.run

        from mathmotion.pipeline import run as pipeline_run
        pipeline_run(
            topic="derivatives",
            config=cfg,
            script=SAMPLE_SCRIPT,
        )

        mock_generate.assert_not_called()


def test_pipeline_writes_files_when_script_provided(tmp_path):
    """Scene files and narration.json must be written when script= is passed."""
    from mathmotion.utils.config import Config
    cfg = MagicMock(spec=Config)
    cfg.manim.default_quality = "draft"
    cfg.storage.jobs_dir = str(tmp_path)
    cfg.llm.repair_max_retries = 0
    cfg.tts.engine = "kokoro"
    cfg.tts.kokoro.voice = "af_heart"

    written_narration = {}

    def fake_tts(script, job_dir, config, engine):
        # Simulate TTS writing actual_duration into narration.json
        import json as _json
        data = _json.loads((job_dir / "narration.json").read_text())
        for scene in data["scenes"]:
            for seg in scene["narration_segments"]:
                seg["actual_duration"] = 1.0
        (job_dir / "narration.json").write_text(_json.dumps(data))

    with (
        patch("mathmotion.pipeline.get_provider"),
        patch("mathmotion.pipeline.get_engine"),
        patch("mathmotion.pipeline.tts.run", side_effect=fake_tts),
        patch("mathmotion.pipeline.inject_actual_durations", side_effect=lambda text, _: text),
        patch("mathmotion.pipeline._run_render_repair_loop", return_value={}),
        patch("mathmotion.pipeline.compose.run", return_value=tmp_path / "out.mp4"),
        patch("mathmotion.pipeline.generate.run"),
    ):
        from mathmotion.pipeline import run as pipeline_run
        pipeline_run(
            topic="derivatives",
            config=cfg,
            script=SAMPLE_SCRIPT,
        )

    # Find the job_dir that was created
    job_dirs = list(tmp_path.iterdir())
    assert len(job_dirs) == 1
    job_dir = job_dirs[0]

    assert (job_dir / "narration.json").exists()
    assert (job_dir / "scenes" / "scene_a.py").exists()
    assert "SceneA" in (job_dir / "scenes" / "scene_a.py").read_text()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_upload.py -v
```

Expected: FAIL — `pipeline.run()` does not accept a `script` parameter yet.

**Step 3: Implement the change in `pipeline.run()`**

In `mathmotion/pipeline.py`, update the signature and add the branch:

```python
def run(
    topic: str,
    config: Config,
    quality: Optional[str] = None,
    level: str = "undergraduate",
    voice: Optional[str] = None,
    tts_engine: Optional[str] = None,
    llm_provider: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    script: Optional[GeneratedScript] = None,          # ← add this
) -> Path:
```

Then replace the generate step (currently lines 105–107):

```python
    progress("Preparing script" if script is not None else "Generating script", 10)
    provider = get_provider(config)
    if script is None:
        script = generate.run(topic, job_dir, config, provider, level=level)
    else:
        scenes_dir = job_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        for scene in script.scenes:
            (scenes_dir / f"{scene.id}.py").write_text(scene.manim_code)
        (job_dir / "narration.json").write_text(script.model_dump_json(indent=2))
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_upload.py -v
```

Expected: PASS

**Step 5: Run full test suite to check no regressions**

```bash
pytest --ignore=tests/test_compose.py --ignore=tests/test_litellm_provider.py --ignore=tests/test_qwen3_engine.py -v
```

Expected: all previously-passing tests still pass.

**Step 6: Commit**

```bash
git add mathmotion/pipeline.py tests/test_pipeline_upload.py
git commit -m "feat: add script param to pipeline.run to skip generate stage"
```

---

### Task 2: Add `POST /api/generate-from-script` endpoint

**Files:**
- Modify: `api/routes.py`
- Test: `tests/test_api_upload.py`

**Step 1: Write the failing tests**

```python
# tests/test_api_upload.py
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

MINIMAL_SCENE_CODE = """from manim import *

class SceneA(Scene):
    def construct(self):
        self.wait(1)
"""

VALID_SCRIPT = {
    "title": "Test",
    "topic": "derivatives",
    "scenes": [
        {
            "id": "scene_a",
            "class_name": "SceneA",
            "manim_code": MINIMAL_SCENE_CODE,
            "narration_segments": [
                {"id": "seg_1", "text": "Hello world", "cue_offset": 0.0}
            ],
        }
    ],
}


def test_upload_valid_script_returns_job_id():
    with patch("api.routes.threading.Thread") as mock_thread:
        mock_thread.return_value.start = lambda: None
        resp = client.post(
            "/api/generate-from-script",
            files={"file": ("script.json", json.dumps(VALID_SCRIPT), "application/json")},
            data={"quality": "draft"},
        )
    assert resp.status_code == 200
    assert "job_id" in resp.json()


def test_upload_invalid_json_returns_error():
    resp = client.post(
        "/api/generate-from-script",
        files={"file": ("script.json", b"not valid json", "application/json")},
    )
    assert resp.status_code == 400
    assert "Invalid JSON" in resp.json()["detail"]


def test_upload_invalid_schema_returns_error():
    bad_script = {"title": "X"}  # missing required fields
    resp = client.post(
        "/api/generate-from-script",
        files={"file": ("script.json", json.dumps(bad_script), "application/json")},
    )
    assert resp.status_code == 400
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api_upload.py -v
```

Expected: FAIL — endpoint does not exist yet.

**Step 3: Implement the endpoint in `api/routes.py`**

Add imports at the top of `api/routes.py`:

```python
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
```

Then add the endpoint:

```python
@router.post("/generate-from-script")
async def start_generate_from_script(
    file: UploadFile = File(...),
    quality: Optional[str] = Form(None),
    tts_engine: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    llm_provider: Optional[str] = Form(None),
):
    import json as _json
    from mathmotion.schemas.script import GeneratedScript
    from mathmotion.stages.generate import _validate
    from mathmotion.utils.errors import ValidationError

    content = await file.read()
    try:
        data = _json.loads(content)
    except _json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        script = _validate(data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _jobs[job_id] = {
        "status": "running",
        "step": "Starting…",
        "pct": 0,
        "output": None,
        "error": None,
    }

    def _run():
        from mathmotion.utils.config import get_config as _get
        _get.cache_clear()
        cfg = _get()

        try:
            from mathmotion.pipeline import run as run_pipeline

            def on_progress(step: str, pct: int):
                _jobs[job_id]["step"] = step
                _jobs[job_id]["pct"] = pct

            output = run_pipeline(
                topic=script.topic,
                config=cfg,
                quality=quality,
                tts_engine=tts_engine,
                voice=voice,
                llm_provider=llm_provider,
                progress_callback=on_progress,
                script=script,
            )
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["output"] = str(output)
            _jobs[job_id]["pct"] = 100
            _jobs[job_id]["step"] = "Done"
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["step"] = "Failed"

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}
```

Note: The existing `from fastapi import APIRouter` import at the top of `routes.py` must be extended to include `HTTPException, UploadFile, File, Form`.

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api_upload.py -v
```

Expected: PASS

**Step 5: Run full test suite**

```bash
pytest --ignore=tests/test_compose.py --ignore=tests/test_litellm_provider.py --ignore=tests/test_qwen3_engine.py -v
```

Expected: all passing.

**Step 6: Commit**

```bash
git add api/routes.py tests/test_api_upload.py
git commit -m "feat: add POST /api/generate-from-script multipart endpoint"
```

---

### Task 3: Frontend — mode toggle and file upload

**Files:**
- Modify: `static/index.html`

No automated tests for JS; verify manually by opening the UI.

**Step 1: Add the toggle styles**

Inside the `<style>` block, before the closing `</style>`, add:

```css
    .mode-toggle {
      display: flex;
      gap: 0;
      margin-bottom: 1.5rem;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #2d2d4e;
      width: fit-content;
    }

    .mode-btn {
      padding: 0.5rem 1.2rem;
      background: #0f0f1a;
      color: #6b7280;
      border: none;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s, color 0.15s;
    }

    .mode-btn.active {
      background: #7c3aed;
      color: #fff;
    }

    .file-drop {
      width: 100%;
      background: #0f0f1a;
      border: 1.5px dashed #2d2d4e;
      border-radius: 8px;
      color: #6b7280;
      font-size: 0.9rem;
      padding: 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.15s;
    }

    .file-drop:hover, .file-drop.has-file { border-color: #7c3aed; color: #e0e0f0; }

    #file-input { display: none; }
```

**Step 2: Add the toggle and file input HTML**

Inside `.card`, before `<label for="topic">`, add the toggle:

```html
    <div class="mode-toggle">
      <button class="mode-btn active" id="mode-generate" onclick="setMode('generate')">Generate</button>
      <button class="mode-btn" id="mode-upload" onclick="setMode('upload')">Upload Script</button>
    </div>
```

After the topic `<textarea>`, add the file input (initially hidden):

```html
    <div id="upload-group" style="display:none">
      <label>Script JSON</label>
      <div class="file-drop" id="file-drop" onclick="document.getElementById('file-input').click()">
        Click to select a <code>.json</code> file
      </div>
      <input type="file" id="file-input" accept=".json" />
    </div>
```

**Step 3: Add the JS for mode switching and upload submission**

In the `<script>` block, add after the `init()` call:

```javascript
    let currentMode = 'generate';

    function setMode(mode) {
      currentMode = mode;
      document.getElementById('mode-generate').classList.toggle('active', mode === 'generate');
      document.getElementById('mode-upload').classList.toggle('active', mode === 'upload');
      document.getElementById('topic').closest('div') || document.getElementById('topic').parentElement;
      // Toggle visibility
      const topicLabel = document.querySelector('label[for="topic"]');
      const topicTA = document.getElementById('topic');
      const uploadGroup = document.getElementById('upload-group');
      topicLabel.style.display = mode === 'generate' ? '' : 'none';
      topicTA.style.display = mode === 'generate' ? '' : 'none';
      uploadGroup.style.display = mode === 'upload' ? '' : 'none';
    }

    // File picker feedback
    document.getElementById('file-input').addEventListener('change', (e) => {
      const f = e.target.files[0];
      const drop = document.getElementById('file-drop');
      if (f) {
        drop.textContent = f.name;
        drop.classList.add('has-file');
      } else {
        drop.textContent = 'Click to select a .json file';
        drop.classList.remove('has-file');
      }
    });
```

**Step 4: Update the submit handler to handle both modes**

Replace the existing `submit` click listener with:

```javascript
    document.getElementById("submit").addEventListener("click", async () => {
      document.getElementById("submit").disabled = true;
      document.getElementById("status-card").style.display = "block";
      document.getElementById("video-section").style.display = "none";
      document.getElementById("error-msg").style.display = "none";
      setProgress("Starting…", 0);

      const voiceEl = document.getElementById("voice");
      const quality = document.getElementById("quality").value;
      const tts_engine = document.getElementById("tts_engine").value;
      const llm_provider = document.getElementById("llm_provider").value;
      const voice = voiceEl && voiceEl.offsetParent ? voiceEl.value : null;

      let job_id;

      if (currentMode === 'generate') {
        const topic = document.getElementById("topic").value.trim();
        if (!topic) { alert("Please enter a topic."); document.getElementById("submit").disabled = false; return; }

        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ topic, quality, level: document.getElementById("level").value, tts_engine, llm_provider, voice }),
        });
        ({ job_id } = await res.json());

      } else {
        const fileInput = document.getElementById("file-input");
        if (!fileInput.files.length) { alert("Please select a JSON file."); document.getElementById("submit").disabled = false; return; }

        const text = await fileInput.files[0].text();
        try { JSON.parse(text); } catch { alert("File is not valid JSON."); document.getElementById("submit").disabled = false; return; }

        const fd = new FormData();
        fd.append("file", fileInput.files[0]);
        if (quality) fd.append("quality", quality);
        if (tts_engine) fd.append("tts_engine", tts_engine);
        if (llm_provider) fd.append("llm_provider", llm_provider);
        if (voice) fd.append("voice", voice);

        const res = await fetch("/api/generate-from-script", { method: "POST", body: fd });
        const data = await res.json();
        if (data.detail || data.error) {
          const errEl = document.getElementById("error-msg");
          errEl.textContent = "Error: " + (data.detail || data.error);
          errEl.style.display = "block";
          document.getElementById("submit").disabled = false;
          return;
        }
        ({ job_id } = data);
      }

      pollTimer = setInterval(() => poll(job_id), 2000);
    });
```

**Step 5: Remove the old submit listener**

The original `document.getElementById("submit").addEventListener("click", ...)` block (lines ~307-334 in the original file) must be deleted — it is fully replaced by the new handler above.

**Step 6: Manual verification**

1. Start the server: `python app.py`
2. Open `http://localhost:PORT`
3. Verify the toggle appears and switching hides/shows the correct inputs
4. Export a script JSON from a previous job's `narration.json` (in the `jobs/` dir), upload it, and verify a job starts and completes

**Step 7: Commit**

```bash
git add static/index.html
git commit -m "feat: add Upload Script mode toggle and file input to frontend"
```

---

### Task 4: Final integration check

**Step 1: Run the full test suite**

```bash
pytest --ignore=tests/test_compose.py --ignore=tests/test_litellm_provider.py --ignore=tests/test_qwen3_engine.py -v
```

Expected: all passing.

**Step 2: Open a PR**

```bash
gh pr create --title "feat: upload GeneratedScript JSON to skip LLM generation" --body "$(cat <<'EOF'
## Summary
- Adds `POST /api/generate-from-script` multipart endpoint — accepts a JSON file matching the `GeneratedScript` schema plus the same quality/tts/voice/llm options as the existing generate endpoint
- Extends `pipeline.run()` with an optional `script` parameter; when provided, scene files and narration.json are written directly, skipping the LLM generate stage while still running TTS, render, and compose
- Adds a "Generate / Upload Script" toggle to the frontend; upload mode shows a file picker, reads the file client-side, and POSTs as multipart/form-data

## Test plan
- [ ] `pytest tests/test_pipeline_upload.py` passes
- [ ] `pytest tests/test_api_upload.py` passes
- [ ] Full test suite passes (excluding heavy integration tests)
- [ ] Manual: upload a real `narration.json` from a previous job and verify end-to-end
EOF
)"
```
