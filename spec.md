# MathMotion — Technical Specification
**Version:** 2.0  
**Format:** Agent-executable specification  
**Purpose:** Complete implementation reference for an AI coding agent building the MathMotion pipeline

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Layout](#2-repository-layout)
3. [Configuration System](#3-configuration-system)
4. [LLM Provider Abstraction](#4-llm-provider-abstraction)
5. [Stage 1 — Prompt Intake](#5-stage-1--prompt-intake)
6. [Stage 2 — LLM Script Generation](#6-stage-2--llm-script-generation)
7. [Stage 3 — Animation Rendering](#7-stage-3--animation-rendering)
8. [Stage 4 — Local TTS Synthesis](#8-stage-4--local-tts-synthesis)
9. [Stage 5 — A/V Composition](#9-stage-5--av-composition)
10. [Stage 6 — Output Delivery](#10-stage-6--output-delivery)
11. [Orchestrator & Job Management](#11-orchestrator--job-management)
12. [REST API](#12-rest-api)
13. [Error Handling & Retry Policy](#13-error-handling--retry-policy)
14. [Timing Synchronisation](#14-timing-synchronisation)
15. [Sandboxing & Security](#15-sandboxing--security)
16. [Dependencies & Environment](#16-dependencies--environment)
17. [Testing Strategy](#17-testing-strategy)
18. [Implementation Roadmap](#18-implementation-roadmap)

---

## 1. System Overview

MathMotion is a fully automated pipeline that converts a natural-language mathematics topic prompt into a narrated, animated `.mp4` video. It combines:

- **LLM code generation** (multi-provider: OpenRouter, Gemini AI Studio, or Ollama) to produce Manim animation code and a structured narration script
- **ManimCE** to render the animation to video
- **Local neural TTS** (Kokoro or Vibevoice) to synthesise narration audio — fully offline, zero API cost
- **FFmpeg** to composite the final video

### 1.1 Design Principles

- **Provider-agnostic LLM layer** — swap between OpenRouter, Gemini AI Studio, and Ollama via config with no code changes
- **Fully offline TTS** — no per-character billing; no network dependency for audio synthesis
- **Deterministic retry** — every failure mode has a defined retry strategy and fallback
- **Agent-friendly output contract** — all stage I/O is file-based with versioned JSON schemas
- **Stateless workers** — each pipeline run is a self-contained job directory; no shared mutable state

### 1.2 Pipeline Stages

```
User Prompt
    │
    ▼
[Stage 1] Prompt Intake          → validated_prompt.json
    │
    ▼
[Stage 2] LLM Script Generation  → scenes/*.py + narration.json
    │
    ├──────────────────────┐
    ▼                      ▼
[Stage 3] Manim Render    [Stage 4] TTS Synthesis       ← parallel
scenes/render/*.mp4        audio/segments/*.wav
    │                      │
    └──────────┬───────────┘
               ▼
[Stage 5] A/V Composition        → output/final.mp4
               │
               ▼
[Stage 6] Output Delivery        → delivery manifest + URL
```

Stages 3 and 4 are **independent and must run in parallel**.

---

## 2. Repository Layout

```
mathmotion/
├── README.md
├── MATHMOTION_SPEC.md            # this file
├── pyproject.toml
├── requirements.txt
├── config.yaml                   # runtime configuration (see §3)
├── .env.example                  # API key template
│
├── mathmotion/
│   ├── __init__.py
│   ├── cli.py                    # Click CLI entry point
│   ├── orchestrator.py           # Job state machine + stage coordination
│   ├── job.py                    # Job dataclass + persistence
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract LLMProvider
│   │   ├── openrouter.py         # OpenRouter provider
│   │   ├── gemini_studio.py      # Gemini AI Studio (google-generativeai SDK)
│   │   ├── ollama.py             # Ollama local inference
│   │   └── factory.py            # LLMProvider factory from config
│   │
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── intake.py             # Stage 1
│   │   ├── generate.py           # Stage 2
│   │   ├── render.py             # Stage 3
│   │   ├── tts.py                # Stage 4
│   │   ├── compose.py            # Stage 5
│   │   └── deliver.py            # Stage 6
│   │
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract TTSEngine
│   │   ├── kokoro.py             # Kokoro integration
│   │   ├── vibevoice.py          # Vibevoice integration
│   │   └── factory.py            # TTSEngine factory from config
│   │
│   ├── schemas/
│   │   ├── prompt.py             # Pydantic: ValidatedPrompt
│   │   ├── script.py             # Pydantic: GeneratedScript, Scene, NarrationSegment
│   │   ├── sync.py               # Pydantic: SyncManifest, SyncSegment
│   │   └── job.py                # Pydantic: Job, JobStatus, JobConfig
│   │
│   └── utils/
│       ├── ffprobe.py            # Audio duration measurement
│       ├── validation.py         # Manim code AST validation
│       └── logging.py            # Structured logging setup
│
├── prompts/
│   ├── system_prompt.txt         # Master LLM system prompt
│   └── domain_hints/
│       ├── calculus.txt
│       ├── linear_algebra.txt
│       ├── geometry.txt
│       └── topology.txt
│
├── jobs/                         # Runtime job directories (gitignored)
│   └── {job_id}/
│       ├── job.json
│       ├── validated_prompt.json
│       ├── narration.json
│       ├── scenes/
│       │   ├── scene_001.py
│       │   └── render/
│       │       └── scene_001.mp4
│       ├── audio/
│       │   └── segments/
│       │       ├── seg_001_a.wav
│       │       └── seg_001_a.mp3
│       ├── sync_manifest.json
│       └── output/
│           └── final.mp4
│
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app
│   ├── routes/
│   │   ├── generate.py
│   │   ├── jobs.py
│   │   └── voices.py
│   └── middleware/
│       └── auth.py
│
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

---

## 3. Configuration System

### 3.1 `config.yaml` Schema

```yaml
# ─── LLM Provider ────────────────────────────────────────────────────────────
llm:
  # Active provider: "openrouter" | "gemini_studio" | "ollama"
  provider: openrouter

  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}          # env var reference
    model_production: google/gemini-2.5-pro
    model_draft: google/gemini-2.0-flash
    site_url: https://mathmotion.app        # for OpenRouter HTTP-Referer header
    site_name: MathMotion

  gemini_studio:
    api_key: ${GEMINI_API_KEY}
    model_production: gemini-2.5-pro-latest
    model_draft: gemini-2.0-flash
    # Uses google-generativeai SDK, not HTTP directly

  ollama:
    base_url: http://localhost:11434
    model_production: gemma3:27b
    model_draft: gemma3:12b
    # No API key required; must have Ollama running locally

  # Shared LLM parameters
  max_tokens: 8192
  temperature: 0.2
  max_retries: 3
  timeout_seconds: 120

# ─── TTS ─────────────────────────────────────────────────────────────────────
tts:
  # Primary engine: "kokoro" | "vibevoice"
  engine: kokoro
  fallback_engine: vibevoice

  kokoro:
    lang_code: a                  # 'a' = American English
    default_voice: af_heart
    speed: 1.0
    sample_rate: 24000
    available_voices:
      - af_heart
      - af_sky
      - af_bella
      - bm_lewis
      - bm_george

  vibevoice:
    default_voice: neutral_female
    speed: 1.0
    available_voices:
      - neutral_female
      - neutral_male
      - expressive_female

  # Shared TTS parameters
  output_format: mp3
  output_bitrate: 192k
  workers: 4                      # parallel synthesis threads

# ─── Manim ───────────────────────────────────────────────────────────────────
manim:
  default_quality: standard       # draft | standard | high
  background_color: "#1a1a2e"
  timeout_seconds: 300            # per scene
  renderer: cairo                 # cairo | opengl (opengl only for high quality)

  quality_presets:
    draft:
      resolution: "854,480"
      frame_rate: 24
      renderer: cairo
    standard:
      resolution: "1280,720"
      frame_rate: 30
      renderer: cairo
    high:
      resolution: "1920,1080"
      frame_rate: 60
      renderer: opengl

# ─── Composition ─────────────────────────────────────────────────────────────
composition:
  sync_strategy: audio_led        # audio_led | time_stretch
  time_stretch_threshold: 0.15    # max drift ratio before stretch is applied
  background_music: false
  bg_music_volume_db: -18
  output_codec_video: libx264
  output_codec_audio: aac
  output_crf: 18
  output_preset: slow
  movflags: +faststart

# ─── Storage ─────────────────────────────────────────────────────────────────
storage:
  jobs_dir: ./jobs
  backend: local                  # local | s3
  s3:
    bucket: ${S3_BUCKET}
    region: ${S3_REGION}
    access_key: ${AWS_ACCESS_KEY_ID}
    secret_key: ${AWS_SECRET_ACCESS_KEY}

# ─── API ─────────────────────────────────────────────────────────────────────
api:
  host: 0.0.0.0
  port: 8000
  workers: 2
  api_key: ${MATHMOTION_API_KEY}  # set to null to disable auth
```

### 3.2 `.env.example`

```bash
OPENROUTER_API_KEY=sk-or-...
GEMINI_API_KEY=AIza...
MATHMOTION_API_KEY=mm-...
S3_BUCKET=mathmotion-outputs
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### 3.3 Config Loading (`mathmotion/utils/config.py`)

- Load `config.yaml` using `pyyaml`
- Resolve `${VAR}` patterns from environment using `os.environ`
- Validate the resolved config with a `pydantic` model
- Expose a global `get_config()` function that returns a cached `Config` instance
- Raise `ConfigError` on missing required env vars at startup, not at runtime

---

## 4. LLM Provider Abstraction

### 4.1 Base Interface (`mathmotion/llm/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMResponse:
    content: str                  # raw text content of the response
    model: str                    # model name actually used
    input_tokens: int
    output_tokens: int
    provider: str

class LLMProvider(ABC):
    """All providers must implement this interface."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> LLMResponse:
        """Synchronous completion. Raises LLMError on failure."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Health check — returns True if provider is reachable."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model identifier string."""
        ...
```

### 4.2 OpenRouter Provider (`mathmotion/llm/openrouter.py`)

```python
import httpx
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError

class OpenRouterProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm.openrouter
        self._quality = config.llm.get("_quality", "production")
        self._model = (
            self.cfg.model_production
            if self._quality == "production"
            else self.cfg.model_draft
        )

    def complete(self, system_prompt, user_prompt, max_tokens=8192,
                 temperature=0.2, json_mode=True) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "HTTP-Referer": self.cfg.site_url,
            "X-Title": self.cfg.site_name,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = httpx.post(
            f"{self.cfg.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            raise LLMError(f"OpenRouter HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self._model),
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
            provider="openrouter",
        )

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.cfg.base_url}/models",
                          headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                          timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    @property
    def model_name(self) -> str:
        return self._model
```

### 4.3 Gemini AI Studio Provider (`mathmotion/llm/gemini_studio.py`)

```python
import google.generativeai as genai
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError

class GeminiStudioProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm.gemini_studio
        genai.configure(api_key=self.cfg.api_key)
        quality = config.llm.get("_quality", "production")
        self._model_id = (
            self.cfg.model_production
            if quality == "production"
            else self.cfg.model_draft
        )
        self._client = genai.GenerativeModel(
            model_name=self._model_id,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            ),
        )

    def complete(self, system_prompt, user_prompt, max_tokens=8192,
                 temperature=0.2, json_mode=True) -> LLMResponse:
        # Gemini AI Studio combines system + user prompt as a prefixed message
        full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        try:
            response = self._client.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    response_mime_type="application/json" if json_mode else "text/plain",
                ),
            )
        except Exception as e:
            raise LLMError(f"Gemini AI Studio error: {e}") from e

        return LLMResponse(
            content=response.text,
            model=self._model_id,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
            provider="gemini_studio",
        )

    def is_available(self) -> bool:
        try:
            genai.list_models()
            return True
        except Exception:
            return False

    @property
    def model_name(self) -> str:
        return self._model_id
```

### 4.4 Ollama Provider (`mathmotion/llm/ollama.py`)

```python
import httpx, json
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError

class OllamaProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm.ollama
        quality = config.llm.get("_quality", "production")
        self._model = (
            self.cfg.model_production
            if quality == "production"
            else self.cfg.model_draft
        )

    def complete(self, system_prompt, user_prompt, max_tokens=8192,
                 temperature=0.2, json_mode=True) -> LLMResponse:
        payload = {
            "model": self._model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        resp = httpx.post(
            f"{self.cfg.base_url}/api/generate",
            json=payload,
            timeout=300,            # local inference can be slow
        )
        if resp.status_code != 200:
            raise LLMError(f"Ollama HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        return LLMResponse(
            content=data["response"],
            model=data.get("model", self._model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            provider="ollama",
        )

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.cfg.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    @property
    def model_name(self) -> str:
        return self._model
```

### 4.5 Provider Factory (`mathmotion/llm/factory.py`)

```python
from .openrouter import OpenRouterProvider
from .gemini_studio import GeminiStudioProvider
from .ollama import OllamaProvider
from .base import LLMProvider

def get_provider(config) -> LLMProvider:
    provider = config.llm.provider
    match provider:
        case "openrouter":    return OpenRouterProvider(config)
        case "gemini_studio": return GeminiStudioProvider(config)
        case "ollama":        return OllamaProvider(config)
        case _: raise ValueError(f"Unknown LLM provider: {provider!r}. "
                                 f"Valid values: openrouter, gemini_studio, ollama")
```

### 4.6 Provider Capability Matrix

| Capability | OpenRouter | Gemini AI Studio | Ollama |
|---|---|---|---|
| JSON mode enforcement | `response_format` | `response_mime_type` | `format: json` |
| Requires internet | Yes | Yes | No (local) |
| API key required | Yes | Yes | No |
| Best model available | Gemini 2.5 Pro | Gemini 2.5 Pro | gemma3:27b |
| Cost | Per token (OpenRouter rates) | Per token (Google rates) | Free (hardware cost) |
| Recommended for | Production | Production (direct Google) | Air-gapped / dev |

### 4.7 LLM Provider Fallback Chain

If the active provider fails (network error, quota exceeded, `LLMError`):

```python
PROVIDER_FALLBACK_ORDER = ["openrouter", "gemini_studio", "ollama"]

def complete_with_fallback(providers: dict[str, LLMProvider], **kwargs):
    errors = []
    for name in PROVIDER_FALLBACK_ORDER:
        if name in providers and providers[name].is_available():
            try:
                return providers[name].complete(**kwargs)
            except LLMError as e:
                errors.append((name, e))
    raise LLMError(f"All providers failed: {errors}")
```

---

## 5. Stage 1 — Prompt Intake

**File:** `mathmotion/stages/intake.py`  
**Input:** Raw user request (dict or CLI args)  
**Output:** `jobs/{job_id}/validated_prompt.json`

### 5.1 Input Schema

```python
class PromptInput(BaseModel):
    topic: str                      # 10–500 characters
    level: Literal["high_school", "undergraduate", "graduate"] = "undergraduate"
    duration_target: int = 120      # seconds, range 60–600
    voice_id: str = "af_heart"      # TTS voice identifier
    quality: Literal["draft", "standard", "high"] = "standard"
    llm_provider: Optional[str] = None  # override config default
    tts_engine: Optional[str] = None    # override config default
```

### 5.2 Validation Rules

- `topic` must be 10–500 characters and non-empty after stripping
- `topic` must pass a mathematical domain classifier (simple keyword heuristic is acceptable for v1)
- `duration_target` clamped to `[60, 600]` with a warning logged if out of range
- `voice_id` must exist in the configured TTS engine's `available_voices` list
- `quality: high` requires `manim.renderer: opengl`; if cairo is configured, degrade to `standard` with a warning

### 5.3 Output: `validated_prompt.json`

```json
{
  "job_id": "job_a1b2c3d4",
  "topic": "Explain the geometric intuition behind hyperbolic geometry",
  "level": "undergraduate",
  "duration_target": 180,
  "voice_id": "af_heart",
  "quality": "standard",
  "llm_provider": "openrouter",
  "tts_engine": "kokoro",
  "domain_hints": ["geometry"],
  "created_at": "2025-03-08T10:00:00Z"
}
```

### 5.4 Domain Hint Detection

Map keywords in `topic` to filenames in `prompts/domain_hints/`:

```python
DOMAIN_MAP = {
    "calculus": ["derivative", "integral", "limit", "series", "differential"],
    "linear_algebra": ["matrix", "vector", "eigenvalue", "eigenvector", "linear", "transform"],
    "geometry": ["hyperbolic", "curve", "manifold", "topology", "surface", "geodesic"],
    "topology": ["homotopy", "homology", "fundamental group", "knot"],
}
```

Detected domain hints are concatenated into the LLM system prompt in Stage 2.

---

## 6. Stage 2 — LLM Script Generation

**File:** `mathmotion/stages/generate.py`  
**Input:** `validated_prompt.json`  
**Output:** `scenes/scene_NNN.py` (one file per scene), `narration.json`

### 6.1 System Prompt (`prompts/system_prompt.txt`)

The system prompt must instruct the model to:

```
You are an expert mathematical animator and educator. You produce Manim animation 
code and narration scripts that explain mathematical concepts with visual clarity.

OUTPUT CONTRACT:
- Respond with a single JSON object only — no prose, no Markdown fences
- The JSON must conform exactly to the GeneratedScript schema below
- All Manim code must be self-contained: one file, all imports explicit
- Each class name must start with Scene_ and be unique within the response
- Narration segments must be short (1–3 sentences max) for clean TTS synthesis
- Narration text must be spoken language only — no LaTeX, no symbols
  Write "x squared" not "x²", "pi" not "π"
- Each self.wait() that maps to a narration segment must be preceded by: # WAIT:{seg_id}
- Do not import os, sys, subprocess, socket, urllib, requests, or httpx in Manim code
- Do not call open(), exec(), or eval()
```

### 6.2 LLM Output Schema (`GeneratedScript`)

```python
class NarrationSegment(BaseModel):
    id: str                         # e.g. "seg_001_a"
    text: str                       # TTS-ready narration (no symbols)
    cue_offset: float               # seconds from scene start to begin this segment
    estimated_duration: float       # seconds (LLM estimate; replaced in Stage 4)
    actual_duration: Optional[float] = None   # filled by Stage 4
    audio_path: Optional[str] = None          # filled by Stage 4

class Scene(BaseModel):
    id: str                         # e.g. "scene_001"
    class_name: str                 # e.g. "Scene_IntroHyperbolicPlane"
    manim_code: str                 # complete Python source as a string
    estimated_duration: float       # seconds
    narration_segments: list[NarrationSegment]

class GeneratedScript(BaseModel):
    title: str
    topic: str
    total_estimated_duration: float
    scenes: list[Scene]             # ordered list; rendered sequentially
```

### 6.3 Narration JSON (`narration.json`)

`narration.json` is written from the `GeneratedScript` after validation and is the persistent record of all narration data. It is mutated in-place by Stage 4 (to fill `actual_duration` and `audio_path`).

```json
{
  "title": "Hyperbolic Geometry: Curves of Constant Negative Curvature",
  "topic": "...",
  "total_estimated_duration": 185.0,
  "scenes": [
    {
      "id": "scene_001",
      "class_name": "Scene_IntroHyperbolicPlane",
      "manim_code": "from manim import *\nclass Scene_IntroHyperbolicPlane(Scene):\n    ...",
      "estimated_duration": 28.5,
      "narration_segments": [
        {
          "id": "seg_001_a",
          "text": "Imagine a geometry where the rules of Euclid no longer apply.",
          "cue_offset": 0.5,
          "estimated_duration": 3.8,
          "actual_duration": null,
          "audio_path": null
        }
      ]
    }
  ]
}
```

### 6.4 Validation Pipeline

Run these checks in order. On any failure, append the error to the prompt and retry (max `config.llm.max_retries` times).

```
CHECK 1 — JSON parse
  json.loads(response.content)
  → Retry prompt: "Response was not valid JSON. Error: {e}. Respond with valid JSON only."

CHECK 2 — Schema validation
  GeneratedScript.model_validate(parsed)
  → Retry prompt: "Schema validation failed: {pydantic_errors}. Correct the structure."

CHECK 3 — Python syntax per scene
  for scene in script.scenes: ast.parse(scene.manim_code)
  → Retry prompt: "Scene {id} has invalid Python syntax: {e}. Fix the code."

CHECK 4 — Forbidden imports
  for scene in script.scenes: check_forbidden_imports(scene.manim_code)
  → Retry prompt: "Scene {id} contains forbidden import '{name}'. Remove it."

CHECK 5 — Class name consistency
  verify class_name appears in manim_code
  → Retry prompt: "Scene {id}: class_name '{name}' not found in manim_code."

CHECK 6 — Narration content
  all segments have non-empty text
  → Retry prompt: "Narration segment {id} has empty text."
```

After passing all checks, write each scene's `manim_code` to `jobs/{job_id}/scenes/scene_{id}.py`.

### 6.5 Forbidden Imports Checker (`mathmotion/utils/validation.py`)

```python
FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "urllib", "urllib3",
    "requests", "httpx", "aiohttp", "ftplib", "smtplib",
    "shutil", "pathlib", "glob", "tempfile",
}

def check_forbidden_imports(code: str) -> list[str]:
    """Returns list of forbidden module names found. Empty list = pass."""
    tree = ast.parse(code)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.names[0].name if isinstance(node, ast.Import) else (node.module or "")
            root = module.split(".")[0]
            if root in FORBIDDEN_MODULES:
                found.append(root)
    return found
```

### 6.6 OpenRouter API Call Example

```python
response = httpx.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://mathmotion.app",
        "X-Title": "MathMotion",
    },
    json={
        "model": "google/gemini-2.5-pro",
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 8192,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    },
    timeout=120,
)
```

---

## 7. Stage 3 — Animation Rendering

**File:** `mathmotion/stages/render.py`  
**Input:** `scenes/scene_NNN.py` files  
**Output:** `scenes/render/scene_NNN.mp4` (one per scene)

### 7.1 Render Command Template

```python
import subprocess

def render_scene(scene_path: Path, class_name: str, output_dir: Path,
                 quality: str, config) -> Path:
    preset = config.manim.quality_presets[quality]
    cmd = [
        "manim", "render",
        str(scene_path),
        class_name,
        "--output_file", f"{scene_path.stem}.mp4",
        "--media_dir", str(output_dir),
        "--resolution", preset.resolution,
        "--frame_rate", str(preset.frame_rate),
        "--renderer", preset.renderer,
        "--format", "mp4",
        "--disable_caching",
        "--background_color", config.manim.background_color,
        "--write_to_movie",
        "-q" + quality[0],              # -qd, -qs, -qh
    ]
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        timeout=config.manim.timeout_seconds,
    )
    if result.returncode != 0:
        raise RenderError(scene_path.stem, result.stderr)
    return output_dir / f"{scene_path.stem}.mp4"
```

### 7.2 Parallelisation

Render all scenes using `concurrent.futures.ThreadPoolExecutor`. Worker count: `min(len(scenes), os.cpu_count() // 2)`. Each scene renders in its own temp subdirectory to avoid Manim media path collisions.

### 7.3 Error Recovery

On `RenderError`:
1. Extract the last 30 lines of `stderr`
2. Pass back to Stage 2's LLM with: `"Scene {id} failed to render. Error:\n{stderr}\n\nFix manim_code for this scene only."`
3. Re-validate and re-render the failing scene only (not the full script)
4. Retry up to 2 times; on persistent failure substitute a blank title-card scene

### 7.4 Blank Title-Card Fallback Scene

```python
FALLBACK_SCENE_TEMPLATE = '''
from manim import *
class Scene_Fallback_{scene_id}(Scene):
    def construct(self):
        title = Text("{title}", font_size=48)
        self.play(Write(title))
        self.wait({duration})
'''
```

### 7.5 Quality Presets

| Preset | Resolution | FPS | Renderer | Approx. time/scene |
|---|---|---|---|---|
| `draft` | 854×480 | 24 | cairo | 5–15s |
| `standard` | 1280×720 | 30 | cairo | 20–60s |
| `high` | 1920×1080 | 60 | opengl | 60–180s |

---

## 8. Stage 4 — Local TTS Synthesis

**File:** `mathmotion/stages/tts.py`  
**Input:** `narration.json`  
**Output:** `audio/segments/*.wav` and `*.mp3`; updates `narration.json` in place

This stage runs **in parallel with Stage 3**. Both are started concurrently from the orchestrator via `asyncio.gather`.

### 8.1 TTS Base Interface (`mathmotion/tts/base.py`)

```python
from abc import ABC, abstractmethod
from pathlib import Path

class TTSEngine(ABC):

    @abstractmethod
    def load(self) -> None:
        """Load model into memory. Called once at worker startup."""
        ...

    @abstractmethod
    def synthesise(self, text: str, output_path: Path,
                   voice_id: str, speed: float = 1.0) -> float:
        """
        Synthesise text to WAV at output_path.
        Returns actual audio duration in seconds.
        Raises TTSError on failure.
        """
        ...

    @abstractmethod
    def available_voices(self) -> list[str]: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...
```

### 8.2 Kokoro Engine (`mathmotion/tts/kokoro.py`)

```python
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from pathlib import Path
from .base import TTSEngine
from mathmotion.utils.errors import TTSError

class KokoroEngine(TTSEngine):

    def __init__(self, config):
        self.cfg = config.tts.kokoro
        self._pipeline = None

    def load(self) -> None:
        self._pipeline = KPipeline(lang_code=self.cfg.lang_code)

    def synthesise(self, text: str, output_path: Path,
                   voice_id: str = None, speed: float = 1.0) -> float:
        if not self._pipeline:
            self.load()
        voice = voice_id or self.cfg.default_voice
        try:
            chunks = []
            for _, _, audio in self._pipeline(text, voice=voice, speed=speed):
                chunks.append(audio)
            full_audio = np.concatenate(chunks)
            wav_path = output_path.with_suffix(".wav")
            sf.write(str(wav_path), full_audio, self.cfg.sample_rate)
            return len(full_audio) / self.cfg.sample_rate
        except Exception as e:
            raise TTSError(f"Kokoro synthesis failed: {e}") from e

    def available_voices(self) -> list[str]:
        return self.cfg.available_voices

    def is_loaded(self) -> bool:
        return self._pipeline is not None
```

### 8.3 Vibevoice Engine (`mathmotion/tts/vibevoice.py`)

```python
from vibevoice import Synthesizer
from pathlib import Path
from .base import TTSEngine
from mathmotion.utils.errors import TTSError

class VibevoiceEngine(TTSEngine):

    def __init__(self, config):
        self.cfg = config.tts.vibevoice
        self._synth = None

    def load(self) -> None:
        self._synth = Synthesizer(voice=self.cfg.default_voice, speed=self.cfg.speed)

    def synthesise(self, text: str, output_path: Path,
                   voice_id: str = None, speed: float = 1.0) -> float:
        if not self._synth:
            self.load()
        voice = voice_id or self.cfg.default_voice
        try:
            wav_path = output_path.with_suffix(".wav")
            self._synth.save(text=text, output_path=str(wav_path), voice=voice)
            return _measure_duration(wav_path)
        except Exception as e:
            raise TTSError(f"Vibevoice synthesis failed: {e}") from e

    def available_voices(self) -> list[str]:
        return self.cfg.available_voices

    def is_loaded(self) -> bool:
        return self._synth is not None
```

### 8.4 TTS Engine Comparison

| Engine | Quality | Speed (CPU) | Speed (GPU) | Voices | Notes |
|---|---|---|---|---|---|
| Kokoro 0.9 | ★★★★★ | ~1.5× realtime | ~20× realtime | 10 (EN) | Default; best quality |
| Vibevoice | ★★★★☆ | ~3× realtime | ~40× realtime | 20+ (EN/multi) | Faster; good for drafts |

### 8.5 Duration Measurement (`mathmotion/utils/ffprobe.py`)

```python
import subprocess, json
from pathlib import Path

def measure_duration(audio_path: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(audio_path),
    ], capture_output=True, text=True, check=True)
    return float(json.loads(result.stdout)["format"]["duration"])
```

### 8.6 WAV to MP3 Conversion

```python
subprocess.run([
    "ffmpeg", "-y",
    "-i", str(wav_path),
    "-codec:a", "libmp3lame",
    "-b:a", config.tts.output_bitrate,
    "-ar", "44100",
    str(mp3_path),
], check=True, capture_output=True)
```

### 8.7 Parallel Synthesis Loop

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import filelock

def synthesise_all(narration: GeneratedScript, job_dir: Path,
                   engine: TTSEngine, config) -> None:
    segments = [
        (scene.id, seg)
        for scene in narration.scenes
        for seg in scene.narration_segments
    ]
    lock = filelock.FileLock(str(job_dir / "narration.json.lock"))

    def synth_one(scene_id, seg):
        out_path = job_dir / "audio" / "segments" / seg.id
        duration = engine.synthesise(text=seg.text, output_path=out_path)
        mp3_path = out_path.with_suffix(".mp3")
        # wav → mp3 conversion here
        return seg.id, duration, str(mp3_path)

    with ThreadPoolExecutor(max_workers=config.tts.workers) as pool:
        futures = {pool.submit(synth_one, sid, seg): (sid, seg)
                   for sid, seg in segments}
        for future in as_completed(futures):
            seg_id, duration, mp3_path = future.result()
            with lock:
                update_narration_json(job_dir, seg_id, duration, mp3_path)
```

### 8.8 TTS Fallback Chain

If primary engine raises `TTSError`:
1. Log WARNING
2. Attempt with `config.tts.fallback_engine`
3. If fallback also fails: write silence of `estimated_duration` length and log ERROR

---

## 9. Stage 5 — A/V Composition

**File:** `mathmotion/stages/compose.py`  
**Input:** `scenes/render/*.mp4`, `audio/segments/*.mp3`, `narration.json`  
**Output:** `output/final.mp4`

### 9.1 Sync Manifest (`sync_manifest.json`)

Built from the updated `narration.json` after Stage 4 completes:

```python
class SyncSegment(BaseModel):
    id: str
    scene_id: str
    audio_path: str
    start_time: float               # seconds from scene start
    actual_duration: float
    estimated_duration: float
    drift_ratio: float              # abs(actual - estimated) / estimated
    action: Literal["passthrough", "stretch", "pad"]

class SyncManifest(BaseModel):
    segments: list[SyncSegment]
    total_audio_duration: float
    total_video_duration: float
    strategy: str
```

`drift_ratio = abs(actual - estimated) / estimated`  
If `drift_ratio > config.composition.time_stretch_threshold` and `strategy == "time_stretch"`, set `action = "stretch"`.

### 9.2 Sync Strategy: `time_stretch` (fast, single render pass)

```python
def time_stretch_audio(input_path: Path, output_path: Path,
                       actual_duration: float, target_duration: float) -> Path:
    ratio = actual_duration / target_duration   # >1 = speed up audio
    ratio = max(0.5, min(2.0, ratio))           # atempo range: [0.5, 2.0]
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-filter:a", f"atempo={ratio:.4f}",
        str(output_path),
    ], check=True, capture_output=True)
    return output_path
```

> For `|ratio - 1.0| > 0.5` (outside atempo single-filter range), chain two filters: `atempo=X,atempo=Y`.

### 9.3 Sync Strategy: `audio_led` (two render passes, highest quality)

1. After Stage 4, inject `actual_duration` values into each scene's `self.wait()` calls
2. Re-run Stage 3 with updated scene files
3. Concatenate audio segments with no padding

Duration injection uses `# WAIT:{seg_id}` comments placed by the LLM:

```python
import re

def inject_actual_durations(scene_code: str, durations: dict[str, float]) -> str:
    lines = scene_code.split("\n")
    out = []
    for i, line in enumerate(lines):
        m = re.search(r"# WAIT:(\S+)", line)
        if m and (i + 1) < len(lines):
            seg_id = m.group(1)
            if seg_id in durations:
                out.append(line)
                out.append(re.sub(
                    r"self\.wait\(.*?\)",
                    f"self.wait({durations[seg_id]:.3f})",
                    lines[i + 1]
                ))
                continue
        out.append(line)
    return "\n".join(out)
```

### 9.4 Final FFmpeg Composition

```bash
# Step 1: Concatenate scene clips
ffmpeg -f concat -safe 0 -i scene_list.txt -c copy assembled.mp4

# Step 2: Build narration_full.mp3 with silence padding at cue offsets
# (Python generates concat_audio.txt with silence files interleaved)
ffmpeg -f concat -safe 0 -i concat_audio.txt -c copy narration_full.mp3

# Step 3: Final encode
ffmpeg \
  -i assembled.mp4 \
  -i narration_full.mp3 \
  -c:v libx264 -preset slow -crf 18 \
  -c:a aac -b:a 192k \
  -movflags +faststart \
  -map 0:v:0 -map 1:a:0 \
  -shortest \
  output/final.mp4
```

`scene_list.txt` format:
```
file 'scenes/render/scene_001.mp4'
file 'scenes/render/scene_002.mp4'
```

### 9.5 Audio Concat with Silence Padding

```python
def build_audio_concat(manifest: SyncManifest, job_dir: Path) -> Path:
    parts = []
    current_time = 0.0
    for seg in sorted(manifest.segments, key=lambda s: s.start_time):
        gap = seg.start_time - current_time
        if gap > 0.05:
            silence_path = job_dir / "audio" / f"silence_{gap:.3f}s.mp3"
            generate_silence(silence_path, duration=gap)   # ffmpeg anullsrc
            parts.append(str(silence_path))
        parts.append(seg.audio_path)
        current_time = seg.start_time + seg.actual_duration

    concat_list = job_dir / "audio" / "concat_list.txt"
    concat_list.write_text("\n".join(f"file '{p}'" for p in parts))
    output = job_dir / "audio" / "narration_full.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list), "-c", "copy", str(output),
    ], check=True, capture_output=True)
    return output
```

---

## 10. Stage 6 — Output Delivery

**File:** `mathmotion/stages/deliver.py`  
**Input:** `output/final.mp4`  
**Output:** Delivery manifest JSON

### 10.1 Local Backend

Copy `final.mp4` to the static files directory. Return a local file path URL.

### 10.2 S3 Backend

```python
import boto3

def upload_to_s3(video_path: Path, job_id: str, config) -> str:
    s3 = boto3.client("s3",
        region_name=config.storage.s3.region,
        aws_access_key_id=config.storage.s3.access_key,
        aws_secret_access_key=config.storage.s3.secret_key,
    )
    key = f"videos/{job_id}/final.mp4"
    s3.upload_file(str(video_path), config.storage.s3.bucket, key,
                   ExtraArgs={"ContentType": "video/mp4"})
    return f"https://{config.storage.s3.bucket}.s3.amazonaws.com/{key}"
```

### 10.3 Delivery Manifest

```json
{
  "job_id": "job_a1b2c3d4",
  "title": "Hyperbolic Geometry: Curves of Constant Negative Curvature",
  "download_url": "https://...",
  "duration_seconds": 183.4,
  "file_size_bytes": 48203942,
  "quality": "standard",
  "llm_provider": "openrouter",
  "llm_model": "google/gemini-2.5-pro",
  "tts_engine": "kokoro",
  "tts_voice": "af_heart",
  "scene_count": 6,
  "completed_at": "2025-03-08T10:05:32Z"
}
```

---

## 11. Orchestrator & Job Management

**File:** `mathmotion/orchestrator.py`

### 11.1 Job Status Enum

```python
class JobStatus(str, Enum):
    PENDING      = "pending"
    INTAKE       = "intake"
    GENERATING   = "generating"
    RENDERING    = "rendering"      # Stages 3 + 4 in parallel
    SYNTHESISING = "synthesising"   # Stage 4 sub-status
    COMPOSITING  = "compositing"
    DELIVERING   = "delivering"
    COMPLETE     = "complete"
    FAILED       = "failed"
```

### 11.2 Orchestrator Run Loop

```python
async def run_job(job: Job, config: Config) -> Job:
    job_dir = Path(config.storage.jobs_dir) / job.id
    try:
        update_job(job, JobStatus.INTAKE, 5)
        validated = await run_stage(intake.run, job, job_dir, config)

        update_job(job, JobStatus.GENERATING, 15)
        script = await run_stage(generate.run, validated, job_dir, config)

        update_job(job, JobStatus.RENDERING, 30)
        # Stages 3 and 4 run concurrently
        await asyncio.gather(
            render.run(script, job_dir, config),
            tts.run(script, job_dir, config),
        )

        update_job(job, JobStatus.COMPOSITING, 80)
        await run_stage(compose.run, script, job_dir, config)

        update_job(job, JobStatus.DELIVERING, 95)
        manifest = await run_stage(deliver.run, job, job_dir, config)

        update_job(job, JobStatus.COMPLETE, 100, result=manifest)
        return job

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        save_job(job, job_dir)
        raise
```

### 11.3 Job Persistence

Serialise `job.json` after every status change. On worker restart, scan the jobs directory for `PENDING` or interrupted jobs and re-queue them.

---

## 12. REST API

**File:** `api/main.py` (FastAPI)

### 12.1 Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/generate` | Submit a new video generation job |
| `GET` | `/api/v1/jobs/{job_id}` | Poll job status and progress |
| `GET` | `/api/v1/jobs/{job_id}/result` | Delivery manifest once complete |
| `GET` | `/api/v1/jobs/{job_id}/download` | Redirect to video URL |
| `DELETE` | `/api/v1/jobs/{job_id}` | Cancel a queued or running job |
| `GET` | `/api/v1/voices` | List available TTS voices per engine |
| `GET` | `/api/v1/providers` | LLM provider availability + active model names |
| `GET` | `/api/v1/health` | Health check + queue depth |

### 12.2 Request / Response Shapes

**`POST /api/v1/generate` request:**
```json
{
  "topic": "Explain the geometric intuition behind hyperbolic geometry",
  "level": "undergraduate",
  "duration_target": 180,
  "voice_id": "af_heart",
  "quality": "standard",
  "llm_provider": "openrouter"
}
```

**`202 Accepted` response:**
```json
{
  "job_id": "job_a1b2c3d4",
  "status": "pending",
  "poll_url": "/api/v1/jobs/job_a1b2c3d4"
}
```

**`GET /api/v1/jobs/{job_id}` response:**
```json
{
  "job_id": "job_a1b2c3d4",
  "status": "rendering",
  "current_stage": 3,
  "stage_name": "Animation Render",
  "progress_pct": 42,
  "eta_seconds": 87,
  "created_at": "2025-03-08T10:00:00Z",
  "updated_at": "2025-03-08T10:02:14Z",
  "error": null
}
```

**`GET /api/v1/providers` response:**
```json
{
  "active_provider": "openrouter",
  "providers": {
    "openrouter":    { "configured": true,  "available": true,  "model": "google/gemini-2.5-pro" },
    "gemini_studio": { "configured": true,  "available": true,  "model": "gemini-2.5-pro-latest" },
    "ollama":        { "configured": true,  "available": false, "model": "gemma3:27b",
                       "error": "Connection refused at http://localhost:11434" }
  }
}
```

### 12.3 Authentication

If `config.api.api_key` is non-null, require `Authorization: Bearer <key>` on all routes except `/api/v1/health`. Return `401 {"detail": "Unauthorized"}` on failure.

---

## 13. Error Handling & Retry Policy

### 13.1 Custom Exceptions (`mathmotion/utils/errors.py`)

```python
class MathMotionError(Exception): pass
class ConfigError(MathMotionError): pass
class LLMError(MathMotionError): pass
class ValidationError(MathMotionError): pass
class RenderError(MathMotionError):
    def __init__(self, scene_id: str, stderr: str):
        self.scene_id = scene_id
        self.stderr = stderr
        super().__init__(f"Render failed for {scene_id}")
class TTSError(MathMotionError): pass
class CompositionError(MathMotionError): pass
```

### 13.2 Retry Matrix

| Stage | Failure | Retries | Strategy |
|---|---|---|---|
| Stage 2 — JSON parse | `JSONDecodeError` | 3 | Append error, retry full generation |
| Stage 2 — Schema | `ValidationError` | 3 | Append Pydantic errors, retry |
| Stage 2 — Syntax | `SyntaxError` | 3 | Append error + scene id, retry |
| Stage 3 — Render | `RenderError` | 2 | Feed stderr to LLM, regenerate scene only |
| Stage 3 — Timeout | `TimeoutExpired` | 1 | Retry with `draft` quality |
| Stage 4 — TTS primary | `TTSError` | 1 | Switch to fallback engine |
| Stage 4 — TTS fallback | `TTSError` | 0 | Write silence placeholder, log ERROR |
| Stage 5 — ffmpeg | `CalledProcessError` | 1 | Retry with `-preset ultrafast` |
| LLM — all providers | `LLMError` | 1 per provider | Try next provider in fallback order |

---

## 14. Timing Synchronisation

### 14.1 Problem

Manim renders at a duration determined by `self.wait()` values. TTS audio duration is unpredictable. Stage 5 reconciles these independently-generated timelines.

### 14.2 Drift Calculation

```
drift_ratio = |actual_tts_duration - estimated_wait_duration| / estimated_wait_duration
```

Acceptable drift without correction: `< 15%`.

### 14.3 Strategy: `time_stretch`

- One render pass; fast
- Audio segments with `drift_ratio >= 0.15` are stretched via `atempo`
- Acceptable quality for `drift_ratio < 0.30`; above 30% switch to `audio_led` for that segment

### 14.4 Strategy: `audio_led`

- Two render passes; perfect quality
- Stage 4 measures actual TTS durations
- `inject_actual_durations()` rewrites `self.wait()` values (see §9.3)
- Stage 3 re-renders; no audio stretching needed

### 14.5 LLM Comment Convention

The system prompt must instruct the LLM to mark every narration-aligned `self.wait()` with a comment on the preceding line:

```python
# WAIT:seg_001_a
self.wait(3.8)       # ← this value will be replaced by inject_actual_durations()
```

`inject_actual_durations()` replaces the `self.wait()` argument on the line **immediately following** a `# WAIT:{seg_id}` comment.

---

## 15. Sandboxing & Security

All LLM-generated Python is executed via Manim's CLI subprocess. The following isolation measures are **mandatory**:

### 15.1 Subprocess Isolation

```python
subprocess.run(cmd,
    user="mathmotion_sandbox",           # dedicated low-privilege OS user
    env={
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_dir),
        "MANIM_MEDIA_DIR": str(output_dir),
    },
    cwd=str(tmp_dir),
    timeout=config.manim.timeout_seconds,
    capture_output=True,
)
```

### 15.2 Static Analysis (must pass before subprocess execution)

- Forbidden modules check (§6.5)
- No `exec()` or `eval()` calls in generated code
- No `__import__()` calls
- No `open()` calls (check AST for `ast.Call` with `func.id == 'open'`)

### 15.3 Resource Limits (Linux)

```python
import resource

def set_limits():
    resource.setrlimit(resource.RLIMIT_CPU,    (120, 120))           # 2 min CPU
    resource.setrlimit(resource.RLIMIT_AS,     (4 * 1024**3, -1))   # 4 GB RAM
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))             # 64 open files

subprocess.run(cmd, preexec_fn=set_limits, ...)
```

---

## 16. Dependencies & Environment

### 16.1 `requirements.txt`

```
# Core pipeline
manim>=0.18.0
pydantic>=2.0
pyyaml>=6.0
click>=8.0
filelock>=3.13
httpx>=0.27
python-dotenv>=1.0

# LLM providers
google-generativeai>=0.8          # Gemini AI Studio

# TTS
kokoro>=0.9.2
soundfile>=0.12
numpy>=1.26

# Audio processing
pydub>=0.25

# API
fastapi>=0.115
uvicorn[standard]>=0.29

# Storage (optional S3)
boto3>=1.34

# Development
pytest>=8.0
pytest-asyncio>=0.23
ruff>=0.4
```

> **Note on Vibevoice:** Verify the exact PyPI package name at implementation time — it may require installation from a GitHub release or wheel file.

### 16.2 System Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| ffmpeg | 6.0+ | A/V composition, audio conversion |
| ffprobe | 6.0+ (bundled with ffmpeg) | Audio duration measurement |
| LaTeX (TeX Live) | 2022+ | Manim math typesetting |
| Cairo | 1.16+ | Manim cairo renderer |
| Ollama | 0.3+ | Local LLM inference (optional) |

### 16.3 `Dockerfile`

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    texlive-latex-base texlive-fonts-recommended texlive-latex-extra \
    libcairo2-dev libpango1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 17. Testing Strategy

### 17.1 Unit Tests

| Module | What to test |
|---|---|
| `llm/factory.py` | Correct provider instantiated for each config value |
| `llm/openrouter.py` | HTTP payload shape, header presence, `LLMError` on non-200 |
| `llm/gemini_studio.py` | `response_mime_type` set when `json_mode=True` |
| `llm/ollama.py` | `format: json` in payload, timeout propagation |
| `stages/intake.py` | All validation rules, domain hint detection |
| `stages/generate.py` | All 6 validation checks; retry counter increments |
| `utils/validation.py` | Forbidden import detection (positive + negative cases) |
| `stages/compose.py` | Drift calculation, strategy selection, `atempo` clamping |
| `utils/ffprobe.py` | Duration parsing from fixture JSON |

### 17.2 Integration Tests

- **Stage 2 → 3:** Generate scene for "what is pi", render it, assert `.mp4` exists with duration > 0
- **Stage 4:** Synthesise short segment with Kokoro, assert `.mp3` exists and duration matches `ffprobe`
- **Parallel stages:** 2-scene job — assert both Stage 3 and 4 complete within 1.5× the slower stage's time
- **Full pipeline smoke test:** Submit "Explain what a derivative is in 30 seconds", assert `final.mp4` exists

### 17.3 Fixture Data

```
tests/fixtures/
├── valid_script.json                  # valid GeneratedScript with 2 scenes
├── invalid_script_bad_python.json     # syntax error in scene code
├── invalid_script_forbidden_import.json  # contains import subprocess
└── narration_with_actuals.json        # narration.json with actual_duration filled
```

---

## 18. Implementation Roadmap

### Phase 1 — Core Pipeline (Week 1)
- [ ] Repository scaffold per §2
- [ ] Config system (§3): yaml loading, env var resolution, Pydantic validation
- [ ] All three LLM providers + factory + `is_available()` (§4)
- [ ] Stages 1 → 5 sequential, single-threaded, Loose Sync only
- [ ] CLI: `python -m mathmotion.cli generate --topic "..." --quality draft`
- [ ] Smoke test: produce a draft video end-to-end

### Phase 2 — Parallel Execution + Tight Sync (Week 2)
- [ ] Stages 3 + 4 run in parallel via `asyncio.gather`
- [ ] Kokoro engine integration + WAV→MP3 pipeline
- [ ] Vibevoice fallback engine
- [ ] `time_stretch` sync strategy
- [ ] `audio_led` sync strategy (re-render pass + duration injection)
- [ ] All 6 LLM validation checks + retry loop
- [ ] LLM provider fallback chain (§4.7)

### Phase 3 — API & Job Queue (Week 3)
- [ ] FastAPI app with all 8 endpoints (§12)
- [ ] Job persistence + restart recovery (§11.3)
- [ ] Async orchestrator with progress updates
- [ ] `/api/v1/providers` health endpoint
- [ ] S3 delivery backend (§10.2)

### Phase 4 — Hardening (Week 4)
- [ ] Subprocess sandboxing (§15)
- [ ] Full retry matrix (§13.2)
- [ ] Unit and integration test suite (§17)
- [ ] Docker image with complete dependency stack (§16.3)
- [ ] Performance benchmarking per quality preset

### Phase 5 — Production (Weeks 5–6)
- [ ] Web frontend: topic input, voice selector, real-time progress, video player
- [ ] Rate limiting and per-user job quotas
- [ ] Job history and re-generation
- [ ] Structured logging → metrics (stage latency, provider failure rates, token usage)

---

## Appendix A — LLM System Prompt Template

```
You are an expert mathematical animator and educator. Generate a complete Manim 
animation script and narration for the following topic.

TOPIC: {topic}
LEVEL: {level}
TARGET DURATION: {duration_target} seconds

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
6. Keep segments short: 1–3 sentences, 3–6 seconds each.
7. Every narration-aligned self.wait() must be preceded by: # WAIT:{segment_id}
8. Forbidden imports: os, sys, subprocess, socket, urllib, requests, httpx.
9. Forbidden calls: open(), exec(), eval(), __import__().
10. self.wait() values are estimates — they will be replaced automatically.
```

---

## Appendix B — Manim Scene Template

Reference structure showing `# WAIT:` comment convention:

```python
from manim import *

class Scene_ExampleScene(Scene):
    def construct(self):
        title = Text("Hyperbolic Geometry", font_size=48)
        self.play(Write(title))
        # WAIT:seg_001_a
        self.wait(3.8)

        self.play(FadeOut(title))

        plane = NumberPlane()
        self.play(Create(plane))
        # WAIT:seg_001_b
        self.wait(5.2)

        circle = Circle(radius=2, color=BLUE)
        self.play(Create(circle))
        # WAIT:seg_001_c
        self.wait(4.0)

        self.play(FadeOut(plane), FadeOut(circle))
```

---

*End of specification — all section references (§N.N) are stable and cross-linked.*
