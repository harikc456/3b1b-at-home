# Remove Duration Input Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the user-facing duration control and let the LLM determine video length based on topic complexity and education level.

**Architecture:** Strip `duration` from the API request model, pipeline signature, and generate stage, then update the system prompt to instruct the LLM to self-determine appropriate length. Remove the Duration UI element from the frontend.

**Tech Stack:** FastAPI, Pydantic, Python, HTML/JS

---

### Task 1: Update the system prompt

**Files:**
- Modify: `prompts/system_prompt.txt`

**Step 1: Replace the TARGET DURATION line**

Open `prompts/system_prompt.txt`. The current line 6 reads:
```
TARGET DURATION: {duration_target} seconds
```

Replace it with:
```
DURATION POLICY: Do NOT target a fixed duration. Determine the appropriate length
based on the topic's complexity and the student's LEVEL. High school explanations
need more scaffolding and analogies; graduate content can be denser and faster.
Teach the concept completely — no artificial padding or premature truncation.
```

Also remove the `{duration_target}` placeholder entirely — it must not appear anywhere in the file after this change (it was the root of a previous KeyError bug).

**Step 2: Verify no leftover placeholder**

Run:
```bash
grep -n "duration_target" prompts/system_prompt.txt
```
Expected: no output (zero matches).

**Step 3: Commit**

```bash
git add prompts/system_prompt.txt
git commit -m "feat: replace fixed duration target with adaptive length policy in prompt"
```

---

### Task 2: Remove duration from generate stage

**Files:**
- Modify: `mathmotion/stages/generate.py:59-83`

**Step 1: Remove the `duration` parameter and its usage**

In `generate.py`, the `run()` function signature at line 65 is:
```python
def run(
    topic: str,
    job_dir: Path,
    config,
    provider: LLMProvider,
    level: str = "undergraduate",
    duration: int = 120,
) -> GeneratedScript:
```

Remove `duration: int = 120,` so it becomes:
```python
def run(
    topic: str,
    job_dir: Path,
    config,
    provider: LLMProvider,
    level: str = "undergraduate",
) -> GeneratedScript:
```

**Step 2: Remove the duration_target format argument**

Find the `.format(...)` call around line 77–83:
```python
system_prompt = Path("prompts/system_prompt.txt").read_text().format(
    topic=topic,
    level=level,
    duration_target=duration,
    domain_hints=domain_hints,
    schema_json=schema_json,
)
```

Remove `duration_target=duration,`:
```python
system_prompt = Path("prompts/system_prompt.txt").read_text().format(
    topic=topic,
    level=level,
    domain_hints=domain_hints,
    schema_json=schema_json,
)
```

**Step 3: Verify**

```bash
grep -n "duration" mathmotion/stages/generate.py
```
Expected: zero matches (the word `duration` should not appear in this file anymore).

**Step 4: Commit**

```bash
git add mathmotion/stages/generate.py
git commit -m "feat: remove duration parameter from generate stage"
```

---

### Task 3: Remove duration from pipeline

**Files:**
- Modify: `mathmotion/pipeline.py:15-53`

**Step 1: Remove the parameter from `run()`**

Current signature at line 15–25:
```python
def run(
    topic: str,
    config: Config,
    quality: Optional[str] = None,
    level: str = "undergraduate",
    duration: int = 120,
    voice: Optional[str] = None,
    tts_engine: Optional[str] = None,
    llm_provider: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Path:
```

Remove `duration: int = 120,`:
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
) -> Path:
```

**Step 2: Remove duration from the generate.run() call**

Current call at line 53:
```python
script = generate.run(topic, job_dir, config, provider, level=level, duration=duration)
```

Update to:
```python
script = generate.run(topic, job_dir, config, provider, level=level)
```

**Step 3: Verify**

```bash
grep -n "duration" mathmotion/pipeline.py
```
Expected: zero matches.

**Step 4: Commit**

```bash
git add mathmotion/pipeline.py
git commit -m "feat: remove duration parameter from pipeline"
```

---

### Task 4: Remove duration from the API route

**Files:**
- Modify: `api/routes.py`

**Step 1: Remove from the request model**

Current `GenerateRequest` at lines 16–23:
```python
class GenerateRequest(BaseModel):
    topic: str
    quality: str = "standard"
    level: str = "undergraduate"
    duration: int = 120
    voice: Optional[str] = None
    tts_engine: Optional[str] = None
    llm_provider: Optional[str] = None
```

Remove `duration: int = 120`:
```python
class GenerateRequest(BaseModel):
    topic: str
    quality: str = "standard"
    level: str = "undergraduate"
    voice: Optional[str] = None
    tts_engine: Optional[str] = None
    llm_provider: Optional[str] = None
```

**Step 2: Remove from the pipeline call**

Current call inside `_run()` around line 53–63:
```python
output = run_pipeline(
    topic=req.topic,
    config=cfg,
    quality=req.quality,
    level=req.level,
    duration=req.duration,
    voice=req.voice,
    tts_engine=req.tts_engine,
    llm_provider=req.llm_provider,
    progress_callback=on_progress,
)
```

Remove `duration=req.duration,`:
```python
output = run_pipeline(
    topic=req.topic,
    config=cfg,
    quality=req.quality,
    level=req.level,
    voice=req.voice,
    tts_engine=req.tts_engine,
    llm_provider=req.llm_provider,
    progress_callback=on_progress,
)
```

**Step 3: Verify**

```bash
grep -n "duration" api/routes.py
```
Expected: zero matches.

**Step 4: Commit**

```bash
git add api/routes.py
git commit -m "feat: remove duration from API request model and pipeline call"
```

---

### Task 5: Remove the Duration control from the frontend

**Files:**
- Modify: `static/index.html`

**Step 1: Remove the Duration `<div>` block**

Find and remove this block (around lines 204–212):
```html
      <div>
        <label for="duration">Duration</label>
        <select id="duration">
          <option value="60">1 minute</option>
          <option value="120" selected>2 minutes</option>
          <option value="180">3 minutes</option>
          <option value="300">5 minutes</option>
        </select>
      </div>
```

The `<div class="grid-2">` wrapper at line 195 now only holds the Level control. Change `grid-2` to remove the two-column layout (just delete the wrapper div or change the class to match a single item — whichever keeps the existing layout consistent with nearby controls).

**Step 2: Remove duration from the fetch body**

Find around line 332:
```js
duration:     parseInt(document.getElementById("duration").value),
```
Delete that line entirely.

**Step 3: Verify**

```bash
grep -n "duration" static/index.html
```
Expected: zero matches.

**Step 4: Commit**

```bash
git add static/index.html
git commit -m "feat: remove Duration selector from UI"
```

---

### Task 6: Final smoke test & PR

**Step 1: Start the app and confirm UI has no Duration field**

```bash
python app.py
```
Open the browser, confirm the Duration dropdown is gone.

**Step 2: Submit a generation job**

Pick any topic, submit, and verify:
- Job progresses past 10% (no `KeyError`)
- Job completes successfully

**Step 3: Create the PR**

```bash
git push -u origin <branch-name>
gh pr create \
  --title "feat: remove duration input — let LLM decide video length" \
  --body "Removes user-facing duration control. The LLM now determines length based on topic complexity and education level."
```
