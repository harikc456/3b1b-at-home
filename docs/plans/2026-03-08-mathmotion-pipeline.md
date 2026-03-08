# MathMotion Pipeline — Personal Use Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Web app (local) where a user types a math topic and gets a narrated `.mp4` video. Personal use — single machine, no auth, no Docker.

**Architecture:** FastAPI backend + single HTML page frontend. Linear pipeline: Prompt → LLM generates Manim code + narration → Render + TTS in parallel → FFmpeg compose → video served in browser.

**Tech Stack:** Python 3.13, uv, FastAPI + Uvicorn, Manim CE, Kokoro TTS, Vibevoice TTS, FFmpeg, Pydantic v2, HTTPX, PyYAML, google-generativeai

**Cuts vs full spec (personal use):**
- No auth, no S3, no Docker, no rate limiting
- No subprocess sandboxing (personal machine)
- No job queue — one job at a time is fine
- Simple HTML/CSS/JS — no React/bundlers
- Synchronous pipeline (no asyncio complexity)

---

## Task 1: Scaffold + Dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `config.yaml`
- Create: `.env.example`
- Create all `__init__.py` files

**Step 1: Install dependencies**

```bash
uv add pydantic pyyaml httpx python-dotenv filelock
uv add fastapi uvicorn
uv add manim
uv add kokoro soundfile numpy
uv add google-generativeai
uv add --dev pytest ruff
```

> **Note on Vibevoice:** Check the PyPI package name before installing — it may be `vibevoice`, `vibe-voice`, or installed from a wheel. Install it when confirmed:
> ```bash
> uv add vibevoice   # adjust name as needed
> ```

**Step 2: Create directory structure**

```bash
mkdir -p mathmotion/{stages,llm,tts,schemas,utils}
mkdir -p prompts/domain_hints
mkdir -p static tests jobs
touch mathmotion/__init__.py
touch mathmotion/stages/__init__.py
touch mathmotion/llm/__init__.py
touch mathmotion/tts/__init__.py
touch mathmotion/schemas/__init__.py
touch mathmotion/utils/__init__.py
touch tests/__init__.py
```

**Step 3: Update `pyproject.toml`**

```toml
[project]
name = "3b1b-at-home"
version = "0.1.0"
description = "Automated math education video pipeline"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "httpx>=0.27",
    "python-dotenv>=1.0",
    "filelock>=3.13",
    "fastapi>=0.115",
    "uvicorn>=0.29",
    "manim>=0.18.0",
    "kokoro>=0.9.2",
    "soundfile>=0.12",
    "numpy>=1.26",
    "google-generativeai>=0.8",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

**Step 4: Create `config.yaml`**

```yaml
llm:
  provider: gemini              # gemini | openrouter | ollama

  gemini:
    api_key: ${GEMINI_API_KEY}
    model: gemini-2.5-pro-latest

  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
    model: google/gemini-2.5-pro
    site_url: https://mathmotion.local
    site_name: MathMotion

  ollama:
    base_url: http://localhost:11434
    model: gemma3:27b

  max_tokens: 8192
  temperature: 0.2
  max_retries: 3
  timeout_seconds: 120

tts:
  engine: kokoro                # kokoro | vibevoice

  kokoro:
    lang_code: a
    voice: af_heart
    speed: 1.0
    sample_rate: 24000
    available_voices:
      - af_heart
      - af_sky
      - af_bella
      - bm_lewis
      - bm_george

  vibevoice:
    voice: neutral_female
    speed: 1.0
    available_voices:
      - neutral_female
      - neutral_male
      - expressive_female

manim:
  default_quality: standard     # draft | standard | high
  background_color: "#1a1a2e"
  timeout_seconds: 300

composition:
  sync_strategy: time_stretch   # time_stretch | audio_led
  time_stretch_threshold: 0.15
  output_crf: 18
  output_preset: slow

storage:
  jobs_dir: ./jobs

server:
  host: 127.0.0.1
  port: 8000
```

**Step 5: Create `.env.example`**

```bash
GEMINI_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-...    # optional, only if using openrouter provider
```

**Step 6: Create `prompts/system_prompt.txt`**

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
6. Keep narration segments short: 1–3 sentences, 3–6 seconds each.
7. Every narration-aligned self.wait() must be preceded by: # WAIT:{segment_id}
8. Forbidden imports: os, sys, subprocess, socket, urllib, requests, httpx.
9. Forbidden calls: open(), exec(), eval(), __import__().
10. self.wait() values are estimates — they will be replaced automatically.
```

**Step 7: Create domain hint files**

```bash
cat > prompts/domain_hints/calculus.txt << 'EOF'
Focus on derivatives, integrals, limits, series, and differential equations.
Use visual representations of areas under curves, slope tangents, and convergence.
EOF

cat > prompts/domain_hints/linear_algebra.txt << 'EOF'
Focus on matrices, vectors, eigenvalues, linear transformations, and vector spaces.
Emphasize geometric intuition of transformations.
EOF

cat > prompts/domain_hints/geometry.txt << 'EOF'
Focus on curves, surfaces, manifolds, geodesics, and curvature.
Visualize spaces and their properties geometrically.
EOF

cat > prompts/domain_hints/topology.txt << 'EOF'
Focus on homotopy, homology, fundamental groups, and knot theory.
Emphasize deformation and connectivity concepts.
EOF
```

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: project scaffold with config for Gemini, OpenRouter, Ollama, Kokoro, Vibevoice"
```

---

## Task 2: Config System + Errors

**Files:**
- Create: `mathmotion/utils/errors.py`
- Create: `mathmotion/utils/config.py`

**Step 1: Create `mathmotion/utils/errors.py`**

```python
class MathMotionError(Exception):
    pass

class ConfigError(MathMotionError):
    pass

class LLMError(MathMotionError):
    pass

class RenderError(MathMotionError):
    def __init__(self, scene_id: str, stderr: str):
        self.scene_id = scene_id
        self.stderr = stderr
        super().__init__(f"Render failed for {scene_id}: {stderr[-500:]}")

class TTSError(MathMotionError):
    pass

class ValidationError(MathMotionError):
    pass

class CompositionError(MathMotionError):
    pass
```

**Step 2: Create `mathmotion/utils/config.py`**

```python
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from .errors import ConfigError

load_dotenv()


class GeminiConfig(BaseModel):
    api_key: str
    model: str


class OpenRouterConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    site_url: str
    site_name: str


class OllamaConfig(BaseModel):
    base_url: str
    model: str


class LLMConfig(BaseModel):
    provider: str
    gemini: GeminiConfig
    openrouter: OpenRouterConfig
    ollama: OllamaConfig
    max_tokens: int = 8192
    temperature: float = 0.2
    max_retries: int = 3
    timeout_seconds: int = 120


class KokoroConfig(BaseModel):
    lang_code: str = "a"
    voice: str = "af_heart"
    speed: float = 1.0
    sample_rate: int = 24000
    available_voices: list[str] = ["af_heart"]


class VibevoiceConfig(BaseModel):
    voice: str = "neutral_female"
    speed: float = 1.0
    available_voices: list[str] = ["neutral_female"]


class TTSConfig(BaseModel):
    engine: str = "kokoro"
    kokoro: KokoroConfig
    vibevoice: VibevoiceConfig


class ManimConfig(BaseModel):
    default_quality: str = "standard"
    background_color: str = "#1a1a2e"
    timeout_seconds: int = 300


class CompositionConfig(BaseModel):
    sync_strategy: str = "time_stretch"
    time_stretch_threshold: float = 0.15
    output_crf: int = 18
    output_preset: str = "slow"


class StorageConfig(BaseModel):
    jobs_dir: str = "./jobs"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Config(BaseModel):
    llm: LLMConfig
    tts: TTSConfig
    manim: ManimConfig
    composition: CompositionConfig
    storage: StorageConfig
    server: ServerConfig


def _resolve_env(obj):
    if isinstance(obj, str):
        def replace(m):
            val = os.environ.get(m.group(1))
            if val is None:
                raise ConfigError(f"Missing env var: {m.group(1)}")
            return val
        return re.sub(r'\$\{(\w+)\}', replace, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(i) for i in obj]
    return obj


@lru_cache(maxsize=1)
def get_config(config_path: str = "config.yaml") -> Config:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config not found: {config_path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(_resolve_env(raw))
```

**Step 3: Quick sanity test**

```python
# tests/test_config.py
def test_config_loads(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    from mathmotion.utils.config import get_config
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.llm.provider in ("gemini", "openrouter", "ollama")
    assert cfg.tts.engine in ("kokoro", "vibevoice")
```

```bash
uv run pytest tests/test_config.py -v
```

**Step 4: Commit**

```bash
git add mathmotion/utils/ tests/test_config.py
git commit -m "feat: config system with Gemini, OpenRouter, Ollama, Kokoro, Vibevoice"
```

---

## Task 3: Pydantic Schema

**Files:**
- Create: `mathmotion/schemas/script.py`

**Step 1: Create `mathmotion/schemas/script.py`**

```python
from typing import Optional
from pydantic import BaseModel


class NarrationSegment(BaseModel):
    id: str
    text: str
    cue_offset: float
    estimated_duration: float
    actual_duration: Optional[float] = None
    audio_path: Optional[str] = None


class Scene(BaseModel):
    id: str
    class_name: str
    manim_code: str
    estimated_duration: float
    narration_segments: list[NarrationSegment]


class GeneratedScript(BaseModel):
    title: str
    topic: str
    total_estimated_duration: float
    scenes: list[Scene]
```

**Step 2: Commit**

```bash
git add mathmotion/schemas/
git commit -m "feat: GeneratedScript Pydantic schema"
```

---

## Task 4: LLM Providers (Gemini + OpenRouter + Ollama)

**Files:**
- Create: `mathmotion/llm/base.py`
- Create: `mathmotion/llm/gemini.py`
- Create: `mathmotion/llm/openrouter.py`
- Create: `mathmotion/llm/ollama.py`
- Create: `mathmotion/llm/factory.py`

**Step 1: Create `mathmotion/llm/base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 8192, temperature: float = 0.2) -> LLMResponse: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...
```

**Step 2: Create `mathmotion/llm/gemini.py`**

```python
import google.generativeai as genai
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError


class GeminiProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm.gemini
        genai.configure(api_key=self.cfg.api_key)
        self._client = genai.GenerativeModel(
            model_name=self.cfg.model,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            ),
        )

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 8192, temperature: float = 0.2) -> LLMResponse:
        full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        try:
            response = self._client.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
        except Exception as e:
            raise LLMError(f"Gemini error: {e}") from e

        return LLMResponse(
            content=response.text,
            model=self.cfg.model,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )

    @property
    def model_name(self) -> str:
        return self.cfg.model
```

**Step 3: Create `mathmotion/llm/openrouter.py`**

```python
import httpx
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError


class OpenRouterProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm.openrouter

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 8192, temperature: float = 0.2) -> LLMResponse:
        try:
            resp = httpx.post(
                f"{self.cfg.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.cfg.api_key}",
                    "HTTP-Referer": self.cfg.site_url,
                    "X-Title": self.cfg.site_name,
                },
                json={
                    "model": self.cfg.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                },
                timeout=120,
            )
        except httpx.RequestError as e:
            raise LLMError(f"OpenRouter network error: {e}") from e

        if resp.status_code != 200:
            raise LLMError(f"OpenRouter {resp.status_code}: {resp.text}")

        data = resp.json()
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.cfg.model),
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
        )

    @property
    def model_name(self) -> str:
        return self.cfg.model
```

**Step 4: Create `mathmotion/llm/ollama.py`**

```python
import httpx
from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError


class OllamaProvider(LLMProvider):
    def __init__(self, config):
        self.cfg = config.llm.ollama

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 8192, temperature: float = 0.2) -> LLMResponse:
        try:
            resp = httpx.post(
                f"{self.cfg.base_url}/api/generate",
                json={
                    "model": self.cfg.model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=300,
            )
        except httpx.RequestError as e:
            raise LLMError(f"Ollama network error: {e}") from e

        if resp.status_code != 200:
            raise LLMError(f"Ollama {resp.status_code}: {resp.text}")

        data = resp.json()
        return LLMResponse(
            content=data["response"],
            model=data.get("model", self.cfg.model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    @property
    def model_name(self) -> str:
        return self.cfg.model
```

**Step 5: Create `mathmotion/llm/factory.py`**

```python
from .base import LLMProvider
from .gemini import GeminiProvider
from .openrouter import OpenRouterProvider
from .ollama import OllamaProvider


def get_provider(config) -> LLMProvider:
    match config.llm.provider:
        case "gemini":     return GeminiProvider(config)
        case "openrouter": return OpenRouterProvider(config)
        case "ollama":     return OllamaProvider(config)
        case p: raise ValueError(f"Unknown LLM provider: {p!r}")
```

**Step 6: Commit**

```bash
git add mathmotion/llm/
git commit -m "feat: LLM providers — Gemini, OpenRouter, Ollama"
```

---

## Task 5: TTS Engines (Kokoro + Vibevoice)

**Files:**
- Create: `mathmotion/tts/base.py`
- Create: `mathmotion/tts/kokoro.py`
- Create: `mathmotion/tts/vibevoice.py`
- Create: `mathmotion/tts/factory.py`

**Step 1: Create `mathmotion/tts/base.py`**

```python
from abc import ABC, abstractmethod
from pathlib import Path


class TTSEngine(ABC):
    @abstractmethod
    def synthesise(self, text: str, output_path: Path, voice: str, speed: float) -> float:
        """Synthesise text → WAV at output_path. Returns actual duration in seconds."""
        ...

    @abstractmethod
    def available_voices(self) -> list[str]: ...
```

**Step 2: Create `mathmotion/tts/kokoro.py`**

```python
import numpy as np
import soundfile as sf
from pathlib import Path
from .base import TTSEngine
from mathmotion.utils.errors import TTSError

_pipeline = None


def _get_pipeline(lang_code: str):
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code=lang_code)
    return _pipeline


class KokoroEngine(TTSEngine):
    def __init__(self, config):
        self.cfg = config.tts.kokoro

    def synthesise(self, text: str, output_path: Path, voice: str = None, speed: float = 1.0) -> float:
        pipeline = _get_pipeline(self.cfg.lang_code)
        voice = voice or self.cfg.voice
        try:
            chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
            audio = np.concatenate(chunks)
            wav_path = output_path.with_suffix(".wav")
            sf.write(str(wav_path), audio, self.cfg.sample_rate)
            return len(audio) / self.cfg.sample_rate
        except Exception as e:
            raise TTSError(f"Kokoro failed: {e}") from e

    def available_voices(self) -> list[str]:
        return list(self.cfg.available_voices)
```

**Step 3: Create `mathmotion/tts/vibevoice.py`**

> **Note:** Verify the Vibevoice import and API before writing this. The PyPI package name and API may differ from the spec. Adjust the `load()` call and `synthesise()` implementation to match the actual library API.

```python
from pathlib import Path
from .base import TTSEngine
from mathmotion.utils.errors import TTSError
from mathmotion.utils.ffprobe import measure_duration


class VibevoiceEngine(TTSEngine):
    def __init__(self, config):
        self.cfg = config.tts.vibevoice
        self._synth = None

    def _load(self):
        from vibevoice import Synthesizer   # adjust import to actual package
        self._synth = Synthesizer(voice=self.cfg.voice, speed=self.cfg.speed)

    def synthesise(self, text: str, output_path: Path, voice: str = None, speed: float = 1.0) -> float:
        if self._synth is None:
            self._load()
        voice = voice or self.cfg.voice
        try:
            wav_path = output_path.with_suffix(".wav")
            self._synth.save(text=text, output_path=str(wav_path), voice=voice)
            return measure_duration(wav_path)
        except Exception as e:
            raise TTSError(f"Vibevoice failed: {e}") from e

    def available_voices(self) -> list[str]:
        return list(self.cfg.available_voices)
```

**Step 4: Create `mathmotion/tts/factory.py`**

```python
from .base import TTSEngine
from .kokoro import KokoroEngine
from .vibevoice import VibevoiceEngine


def get_engine(config, engine_name: str = None) -> TTSEngine:
    name = engine_name or config.tts.engine
    match name:
        case "kokoro":    return KokoroEngine(config)
        case "vibevoice": return VibevoiceEngine(config)
        case n: raise ValueError(f"Unknown TTS engine: {n!r}")
```

**Step 5: Commit**

```bash
git add mathmotion/tts/
git commit -m "feat: TTS engines — Kokoro and Vibevoice"
```

---

## Task 6: Validation + FFprobe Utils

**Files:**
- Create: `mathmotion/utils/validation.py`
- Create: `mathmotion/utils/ffprobe.py`
- Create: `tests/test_validation.py`

**Step 1: Create `mathmotion/utils/validation.py`**

```python
import ast

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "urllib", "urllib3",
    "requests", "httpx", "aiohttp", "shutil",
}
FORBIDDEN_CALLS = {"eval", "exec", "open", "__import__"}


def check_forbidden_imports(code: str) -> list[str]:
    tree = ast.parse(code)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.names[0].name if isinstance(node, ast.Import) else (node.module or "")
            root = module.split(".")[0]
            if root in FORBIDDEN_MODULES:
                found.append(root)
    return found


def check_forbidden_calls(code: str) -> list[str]:
    tree = ast.parse(code)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                found.append(node.func.id)
    return found
```

**Step 2: Create `mathmotion/utils/ffprobe.py`**

```python
import json
import subprocess
from pathlib import Path


def measure_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])
```

**Step 3: Write tests**

```python
# tests/test_validation.py
from mathmotion.utils.validation import check_forbidden_imports, check_forbidden_calls

def test_detects_subprocess():
    assert "subprocess" in check_forbidden_imports("import subprocess")

def test_clean_code_passes():
    assert check_forbidden_imports("from manim import *") == []

def test_detects_eval():
    assert "eval" in check_forbidden_calls("x = eval('1+1')")
```

```bash
uv run pytest tests/test_validation.py -v
```
Expected: PASS

**Step 4: Commit**

```bash
git add mathmotion/utils/ tests/test_validation.py
git commit -m "feat: AST validation and ffprobe utils"
```

---

## Task 7: Stage 2 — LLM Script Generation

**Files:**
- Create: `mathmotion/stages/generate.py`

**Step 1: Create `mathmotion/stages/generate.py`**

```python
import ast
import json
import logging
from pathlib import Path

from mathmotion.llm.base import LLMProvider
from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import LLMError, ValidationError
from mathmotion.utils.validation import check_forbidden_imports, check_forbidden_calls

logger = logging.getLogger(__name__)

DOMAIN_MAP = {
    "calculus":       ["derivative", "integral", "limit", "series", "differential"],
    "linear_algebra": ["matrix", "vector", "eigenvalue", "eigenvector", "linear", "transform"],
    "geometry":       ["hyperbolic", "curve", "manifold", "surface", "geodesic"],
    "topology":       ["homotopy", "homology", "fundamental group", "knot"],
}


def _detect_domains(topic: str) -> list[str]:
    t = topic.lower()
    return [d for d, kws in DOMAIN_MAP.items() if any(kw in t for kw in kws)]


def _validate(data: dict) -> GeneratedScript:
    try:
        script = GeneratedScript.model_validate(data)
    except Exception as e:
        raise ValidationError(f"Schema error: {e}")

    for scene in script.scenes:
        try:
            ast.parse(scene.manim_code)
        except SyntaxError as e:
            raise ValidationError(f"Scene {scene.id} syntax error: {e}")

        bad = check_forbidden_imports(scene.manim_code)
        if bad:
            raise ValidationError(f"Scene {scene.id} forbidden imports: {bad}")

        bad = check_forbidden_calls(scene.manim_code)
        if bad:
            raise ValidationError(f"Scene {scene.id} forbidden calls: {bad}")

        if scene.class_name not in scene.manim_code:
            raise ValidationError(f"Scene {scene.id}: class '{scene.class_name}' not found in code")

        for seg in scene.narration_segments:
            if not seg.text.strip():
                raise ValidationError(f"Segment {seg.id} has empty text")

    return script


def run(topic: str, job_dir: Path, config, provider: LLMProvider,
        level: str = "undergraduate", duration: int = 120) -> GeneratedScript:
    domains = _detect_domains(topic)
    domain_hints = ""
    for d in domains:
        f = Path(f"prompts/domain_hints/{d}.txt")
        if f.exists():
            domain_hints += f.read_text() + "\n"
    if not domain_hints:
        domain_hints = "General mathematics."

    schema_json = json.dumps(GeneratedScript.model_json_schema(), indent=2)
    system_prompt = Path("prompts/system_prompt.txt").read_text().format(
        topic=topic, level=level, duration_target=duration,
        domain_hints=domain_hints, schema_json=schema_json,
    )

    base_prompt = f"Generate a Manim animation script for: {topic}"
    user_prompt = base_prompt
    last_error = None

    for attempt in range(config.llm.max_retries + 1):
        if attempt > 0:
            logger.warning(f"Retry {attempt}: {last_error}")
            user_prompt = f"{base_prompt}\n\nFix from previous attempt: {last_error}"

        try:
            resp = provider.complete(system_prompt, user_prompt,
                                     config.llm.max_tokens, config.llm.temperature)
        except LLMError as e:
            last_error = str(e)
            continue

        try:
            data = json.loads(resp.content)
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
            continue

        try:
            script = _validate(data)
        except ValidationError as e:
            last_error = str(e)
            continue

        scenes_dir = job_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        for scene in script.scenes:
            (scenes_dir / f"{scene.id}.py").write_text(scene.manim_code)

        (job_dir / "narration.json").write_text(script.model_dump_json(indent=2))
        logger.info(f"Generated {len(script.scenes)} scenes")
        return script

    raise LLMError(f"Failed after {config.llm.max_retries + 1} attempts. Last: {last_error}")
```

**Step 2: Commit**

```bash
git add mathmotion/stages/generate.py
git commit -m "feat: Stage 2 LLM script generation with retry and validation"
```

---

## Task 8: Stage 3 — Manim Render

**Files:**
- Create: `mathmotion/stages/render.py`

**Step 1: Create `mathmotion/stages/render.py`**

```python
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript, Scene
from mathmotion.utils.errors import RenderError

logger = logging.getLogger(__name__)

QUALITY_FLAGS = {
    "draft":    ("-ql", "854,480",   "24"),
    "standard": ("-qm", "1280,720",  "30"),
    "high":     ("-qh", "1920,1080", "60"),
}

FALLBACK_TEMPLATE = '''from manim import *
class Scene_Fallback_{sid}(Scene):
    def construct(self):
        self.play(Write(Text("{title}", font_size=48)))
        self.wait({dur})
'''


def _render(scene: Scene, scenes_dir: Path, render_dir: Path, quality: str, config) -> Path:
    scene_path = scenes_dir / f"{scene.id}.py"
    q_flag, resolution, fps = QUALITY_FLAGS[quality]

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run([
            "manim", "render",
            str(scene_path), scene.class_name,
            "--output_file", f"{scene.id}.mp4",
            "--media_dir", tmp,
            "--resolution", resolution,
            "--frame_rate", fps,
            "--renderer", "cairo",
            "--format", "mp4",
            "--disable_caching",
            "--background_color", config.manim.background_color,
            "--write_to_movie", q_flag,
        ], capture_output=True, text=True, timeout=config.manim.timeout_seconds)

        if result.returncode != 0:
            raise RenderError(scene.id, result.stderr)

        mp4s = list(Path(tmp).rglob("*.mp4"))
        if not mp4s:
            raise RenderError(scene.id, "No .mp4 produced")

        out = render_dir / f"{scene.id}.mp4"
        shutil.copy(mp4s[0], out)
        logger.info(f"Rendered {scene.id}")
        return out


def _fallback(scene: Scene, scenes_dir: Path, render_dir: Path, config) -> Path:
    code = FALLBACK_TEMPLATE.format(
        sid=scene.id,
        title=scene.id.replace("_", " ").title(),
        dur=max(3, int(scene.estimated_duration)),
    )
    fb = Scene(id=scene.id, class_name=f"Scene_Fallback_{scene.id}",
               manim_code=code, estimated_duration=scene.estimated_duration,
               narration_segments=scene.narration_segments)
    (scenes_dir / f"{scene.id}.py").write_text(code)
    return _render(fb, scenes_dir, render_dir, "draft", config)


def run(script: GeneratedScript, job_dir: Path, config) -> dict[str, Path]:
    quality = config.manim.default_quality
    scenes_dir = job_dir / "scenes"
    render_dir = scenes_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    def do(scene: Scene):
        for attempt in range(3):
            try:
                return scene.id, _render(scene, scenes_dir, render_dir, quality, config)
            except (RenderError, subprocess.TimeoutExpired) as e:
                logger.warning(f"Render attempt {attempt+1} failed for {scene.id}: {e}")
        logger.error(f"Using fallback title card for {scene.id}")
        return scene.id, _fallback(scene, scenes_dir, render_dir, config)

    workers = max(1, (os.cpu_count() or 2) // 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(do, script.scenes))
```

**Step 2: Commit**

```bash
git add mathmotion/stages/render.py
git commit -m "feat: Stage 3 Manim render with parallel scenes and fallback"
```

---

## Task 9: Stage 4 — TTS Synthesis

**Files:**
- Create: `mathmotion/stages/tts.py`

**Step 1: Create `mathmotion/stages/tts.py`**

```python
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import filelock

from mathmotion.schemas.script import GeneratedScript, NarrationSegment
from mathmotion.tts.base import TTSEngine
from mathmotion.utils.errors import TTSError

logger = logging.getLogger(__name__)


def _wav_to_mp3(wav_path: Path) -> Path:
    mp3 = wav_path.with_suffix(".mp3")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", str(mp3),
    ], check=True, capture_output=True)
    return mp3


def _silence(path: Path, duration: float) -> Path:
    mp3 = path.with_suffix(".mp3")
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(duration), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3),
    ], check=True, capture_output=True)
    return mp3


def run(script: GeneratedScript, job_dir: Path, config, engine: TTSEngine) -> None:
    audio_dir = job_dir / "audio" / "segments"
    audio_dir.mkdir(parents=True, exist_ok=True)

    tts_cfg = config.tts.kokoro if config.tts.engine == "kokoro" else config.tts.vibevoice
    voice = tts_cfg.voice
    speed = tts_cfg.speed

    segments = [(scene.id, seg) for scene in script.scenes for seg in scene.narration_segments]
    narration_path = job_dir / "narration.json"
    lock = filelock.FileLock(str(narration_path) + ".lock")

    def synth(scene_id: str, seg: NarrationSegment):
        out = audio_dir / seg.id
        try:
            duration = engine.synthesise(seg.text, out, voice=voice, speed=speed)
            mp3 = _wav_to_mp3(out.with_suffix(".wav"))
        except TTSError as e:
            logger.error(f"TTS failed for {seg.id}: {e} — using silence")
            mp3 = _silence(out, seg.estimated_duration)
            duration = seg.estimated_duration
        return seg.id, duration, str(mp3)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(synth, sid, seg): seg for sid, seg in segments}
        for future in as_completed(futures):
            seg_id, duration, mp3_path = future.result()
            with lock:
                data = json.loads(narration_path.read_text())
                for scene in data["scenes"]:
                    for seg in scene["narration_segments"]:
                        if seg["id"] == seg_id:
                            seg["actual_duration"] = duration
                            seg["audio_path"] = mp3_path
                narration_path.write_text(json.dumps(data, indent=2))
            logger.info(f"Synthesised {seg_id} ({duration:.2f}s)")
```

**Step 2: Commit**

```bash
git add mathmotion/stages/tts.py
git commit -m "feat: Stage 4 TTS synthesis with Kokoro/Vibevoice and silence fallback"
```

---

## Task 10: Stage 5 — A/V Composition

**Files:**
- Create: `mathmotion/stages/compose.py`
- Create: `tests/test_compose.py`

**Step 1: Tests for tricky logic**

```python
# tests/test_compose.py
from mathmotion.stages.compose import compute_drift, inject_actual_durations

def test_drift():
    assert abs(compute_drift(3.5, 3.0) - 0.1667) < 0.001

def test_inject_replaces_wait():
    code = "# WAIT:seg_001_a\nself.wait(3.8)"
    assert "self.wait(4.200)" in inject_actual_durations(code, {"seg_001_a": 4.2})

def test_inject_ignores_unknown():
    code = "# WAIT:seg_999\nself.wait(3.0)"
    assert "self.wait(3.0)" in inject_actual_durations(code, {"other": 4.0})
```

```bash
uv run pytest tests/test_compose.py -v
```
Expected: FAIL (module not found yet)

**Step 2: Create `mathmotion/stages/compose.py`**

```python
import json
import logging
import re
import subprocess
from pathlib import Path

from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import CompositionError

logger = logging.getLogger(__name__)


def compute_drift(actual: float, estimated: float) -> float:
    return abs(actual - estimated) / estimated if estimated else 0.0


def inject_actual_durations(code: str, durations: dict[str, float]) -> str:
    lines, out, i = code.split("\n"), [], 0
    while i < len(lines):
        m = re.search(r"# WAIT:(\S+)", lines[i])
        if m and (i + 1) < len(lines) and m.group(1) in durations:
            out.append(lines[i])
            out.append(re.sub(r"self\.wait\(.*?\)",
                               f"self.wait({durations[m.group(1)]:.3f})",
                               lines[i + 1]))
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _silence(path: Path, duration: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(duration), "-codec:a", "libmp3lame", "-b:a", "192k", str(path),
    ], check=True, capture_output=True)


def _build_audio_track(script: GeneratedScript, job_dir: Path) -> Path:
    audio_dir = job_dir / "audio"
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
            current = seg.cue_offset + (seg.actual_duration or seg.estimated_duration)

    concat = audio_dir / "concat.txt"
    concat.write_text("\n".join(f"file '{p}'" for p in parts))
    out = audio_dir / "narration.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat), "-c", "copy", str(out),
    ], check=True, capture_output=True)
    return out


def run(job_dir: Path, config) -> Path:
    script = GeneratedScript.model_validate(
        json.loads((job_dir / "narration.json").read_text())
    )

    threshold = config.composition.time_stretch_threshold
    strategy = config.composition.sync_strategy

    if strategy == "audio_led":
        durations = {
            seg.id: seg.actual_duration
            for scene in script.scenes
            for seg in scene.narration_segments
            if seg.actual_duration is not None
        }
        scenes_dir = job_dir / "scenes"
        for scene in script.scenes:
            sf = scenes_dir / f"{scene.id}.py"
            if sf.exists():
                sf.write_text(inject_actual_durations(sf.read_text(), durations))
        from mathmotion.stages import render
        render.run(script, job_dir, config)

    elif strategy == "time_stretch":
        for scene in script.scenes:
            for seg in scene.narration_segments:
                if not seg.audio_path or not seg.actual_duration:
                    continue
                if compute_drift(seg.actual_duration, seg.estimated_duration) >= threshold:
                    ratio = max(0.5, min(2.0, seg.actual_duration / seg.estimated_duration))
                    inp = Path(seg.audio_path)
                    out = inp.with_stem(inp.stem + "_s")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(inp),
                        "-filter:a", f"atempo={ratio:.4f}", str(out),
                    ], check=True, capture_output=True)
                    seg.audio_path = str(out)
                    seg.actual_duration = seg.estimated_duration

    audio = _build_audio_track(script, job_dir)

    scene_list = job_dir / "scene_list.txt"
    render_dir = job_dir / "scenes" / "render"
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

    logger.info(f"Final: {final}")
    return final
```

**Step 3: Run tests**

```bash
uv run pytest tests/test_compose.py -v
```
Expected: PASS

**Step 4: Commit**

```bash
git add mathmotion/stages/compose.py tests/test_compose.py
git commit -m "feat: Stage 5 A/V composition with time_stretch and audio_led"
```

---

## Task 11: Pipeline Orchestrator

**Files:**
- Create: `mathmotion/pipeline.py`

**Step 1: Create `mathmotion/pipeline.py`**

```python
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mathmotion.llm.factory import get_provider
from mathmotion.tts.factory import get_engine
from mathmotion.stages import generate, render, tts, compose
from mathmotion.utils.config import Config

logger = logging.getLogger(__name__)


def run(
    topic: str,
    config: Config,
    quality: str = None,
    level: str = "undergraduate",
    duration: int = 120,
    voice: str = None,
    tts_engine: str = None,
    llm_provider: str = None,
    progress_callback=None,   # optional: called with (step: str, pct: int)
) -> Path:
    def progress(step: str, pct: int):
        logger.info(f"[{pct}%] {step}")
        if progress_callback:
            progress_callback(step, pct)

    # Apply overrides
    if llm_provider:
        config.llm.provider = llm_provider
    if tts_engine:
        config.tts.engine = tts_engine
    if voice:
        if config.tts.engine == "kokoro":
            config.tts.kokoro.voice = voice
        else:
            config.tts.vibevoice.voice = voice
    if quality:
        config.manim.default_quality = quality

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = Path(config.storage.jobs_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Job {job_id}: {topic!r}")

    progress("Generating script", 10)
    provider = get_provider(config)
    script = generate.run(topic, job_dir, config, provider, level=level, duration=duration)

    progress("Rendering animation + synthesising audio", 30)
    engine = get_engine(config)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rf = pool.submit(render.run, script, job_dir, config)
        tf = pool.submit(tts.run, script, job_dir, config, engine)
        rf.result()
        tf.result()

    progress("Composing video", 80)
    final = compose.run(job_dir, config)

    progress("Done", 100)
    return final
```

**Step 2: Commit**

```bash
git add mathmotion/pipeline.py
git commit -m "feat: pipeline orchestrator with progress callback and overrides"
```

---

## Task 12: Web UI — FastAPI Backend

**Files:**
- Create: `app.py`
- Create: `api/routes.py`

**Step 1: Create `api/routes.py`**

```python
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from mathmotion.pipeline import run as run_pipeline
from mathmotion.utils.config import get_config

router = APIRouter()

# Simple in-memory job tracker (personal use — no persistence needed)
_jobs: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    topic: str
    quality: str = "standard"
    level: str = "undergraduate"
    duration: int = 120
    voice: Optional[str] = None
    tts_engine: Optional[str] = None
    llm_provider: Optional[str] = None


@router.post("/generate")
def start_generate(req: GenerateRequest):
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _jobs[job_id] = {"status": "running", "step": "Starting...", "pct": 0, "output": None, "error": None}

    def _run():
        config = get_config()
        try:
            def on_progress(step, pct):
                _jobs[job_id]["step"] = step
                _jobs[job_id]["pct"] = pct

            output = run_pipeline(
                topic=req.topic,
                config=config,
                quality=req.quality,
                level=req.level,
                duration=req.duration,
                voice=req.voice,
                tts_engine=req.tts_engine,
                llm_provider=req.llm_provider,
                progress_callback=on_progress,
            )
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["output"] = str(output)
            _jobs[job_id]["pct"] = 100
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return job


@router.get("/download/{job_id}")
def download_video(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "complete":
        return {"error": "Video not ready"}
    path = Path(job["output"])
    return FileResponse(path, media_type="video/mp4",
                        filename=f"mathmotion_{job_id}.mp4")


@router.get("/config/options")
def get_options():
    config = get_config()
    return {
        "llm_providers": ["gemini", "openrouter", "ollama"],
        "tts_engines": ["kokoro", "vibevoice"],
        "kokoro_voices": config.tts.kokoro.available_voices,
        "vibevoice_voices": config.tts.vibevoice.available_voices,
        "qualities": ["draft", "standard", "high"],
        "levels": ["high_school", "undergraduate", "graduate"],
        "defaults": {
            "llm_provider": config.llm.provider,
            "tts_engine": config.tts.engine,
            "quality": config.manim.default_quality,
        },
    }
```

**Step 2: Create `app.py`**

```python
import logging
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from mathmotion.utils.config import get_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="MathMotion")
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return Path("static/index.html").read_text()


if __name__ == "__main__":
    config = get_config()
    url = f"http://{config.server.host}:{config.server.port}"
    print(f"\nMathMotion running at {url}\n")
    webbrowser.open(url)
    uvicorn.run("app:app", host=config.server.host, port=config.server.port, reload=False)
```

**Step 3: Commit**

```bash
git add app.py api/routes.py
git commit -m "feat: FastAPI backend with generate, status, download, and config endpoints"
```

---

## Task 13: Web UI — Frontend Page

**Files:**
- Create: `static/index.html`

**Step 1: Create `static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MathMotion</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f0f1a;
      color: #e0e0f0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }

    h1 {
      font-size: 2rem;
      font-weight: 700;
      color: #a78bfa;
      margin-bottom: 0.25rem;
    }

    .subtitle {
      color: #6b7280;
      margin-bottom: 2.5rem;
      font-size: 0.95rem;
    }

    .card {
      background: #1a1a2e;
      border: 1px solid #2d2d4e;
      border-radius: 12px;
      padding: 2rem;
      width: 100%;
      max-width: 680px;
    }

    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 500;
      color: #9ca3af;
      margin-bottom: 0.4rem;
      margin-top: 1.25rem;
    }

    label:first-child { margin-top: 0; }

    textarea, select {
      width: 100%;
      background: #0f0f1a;
      border: 1px solid #2d2d4e;
      border-radius: 8px;
      color: #e0e0f0;
      font-size: 0.95rem;
      padding: 0.65rem 0.85rem;
      outline: none;
      transition: border-color 0.2s;
    }

    textarea { resize: vertical; min-height: 100px; font-family: inherit; }
    textarea:focus, select:focus { border-color: #7c3aed; }

    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }

    #voice-row { display: none; }

    button#submit {
      margin-top: 1.75rem;
      width: 100%;
      padding: 0.85rem;
      background: #7c3aed;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }

    button#submit:hover:not(:disabled) { background: #6d28d9; }
    button#submit:disabled { background: #4b5563; cursor: not-allowed; }

    #status-card {
      display: none;
      margin-top: 1.5rem;
      background: #1a1a2e;
      border: 1px solid #2d2d4e;
      border-radius: 12px;
      padding: 1.5rem;
      width: 100%;
      max-width: 680px;
    }

    .progress-bar-wrap {
      background: #0f0f1a;
      border-radius: 99px;
      height: 10px;
      overflow: hidden;
      margin: 0.75rem 0;
    }

    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, #7c3aed, #a78bfa);
      border-radius: 99px;
      width: 0%;
      transition: width 0.4s ease;
    }

    #step-text { font-size: 0.9rem; color: #9ca3af; }
    #error-text { color: #f87171; font-size: 0.9rem; margin-top: 0.5rem; }

    #video-section { display: none; margin-top: 0.75rem; }

    video {
      width: 100%;
      border-radius: 8px;
      background: #000;
      margin-top: 0.5rem;
    }

    #download-btn {
      display: inline-block;
      margin-top: 0.75rem;
      padding: 0.55rem 1.25rem;
      background: #059669;
      color: #fff;
      border-radius: 8px;
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 600;
    }

    #download-btn:hover { background: #047857; }
  </style>
</head>
<body>
  <h1>MathMotion</h1>
  <p class="subtitle">Type a math topic and get an animated video explanation</p>

  <div class="card">
    <label for="topic">Topic</label>
    <textarea id="topic" placeholder="e.g. Explain what a derivative is and why it matters geometrically"></textarea>

    <div class="row">
      <div>
        <label for="level">Level</label>
        <select id="level">
          <option value="high_school">High School</option>
          <option value="undergraduate" selected>Undergraduate</option>
          <option value="graduate">Graduate</option>
        </select>
      </div>
      <div>
        <label for="duration">Duration (seconds)</label>
        <select id="duration">
          <option value="60">60s</option>
          <option value="120" selected>120s</option>
          <option value="180">180s</option>
          <option value="300">300s</option>
        </select>
      </div>
    </div>

    <div class="row">
      <div>
        <label for="quality">Quality</label>
        <select id="quality">
          <option value="draft">Draft (fast)</option>
          <option value="standard" selected>Standard</option>
          <option value="high">High (slow)</option>
        </select>
      </div>
      <div>
        <label for="llm_provider">LLM Provider</label>
        <select id="llm_provider"></select>
      </div>
    </div>

    <div class="row">
      <div>
        <label for="tts_engine">Voice Engine</label>
        <select id="tts_engine"></select>
      </div>
      <div id="voice-row">
        <label for="voice">Voice</label>
        <select id="voice"></select>
      </div>
    </div>

    <button id="submit">Generate Video</button>
  </div>

  <div id="status-card">
    <div id="step-text">Starting…</div>
    <div class="progress-bar-wrap">
      <div class="progress-bar" id="progress-bar"></div>
    </div>
    <div id="error-text"></div>
    <div id="video-section">
      <video id="video-player" controls></video>
      <br>
      <a id="download-btn" download>Download MP4</a>
    </div>
  </div>

  <script>
    let options = {};
    let pollInterval = null;

    async function loadOptions() {
      const res = await fetch("/api/config/options");
      options = await res.json();

      const llmSel = document.getElementById("llm_provider");
      options.llm_providers.forEach(p => {
        const o = document.createElement("option");
        o.value = p;
        o.textContent = p.charAt(0).toUpperCase() + p.slice(1);
        if (p === options.defaults.llm_provider) o.selected = true;
        llmSel.appendChild(o);
      });

      const ttsSel = document.getElementById("tts_engine");
      options.tts_engines.forEach(e => {
        const o = document.createElement("option");
        o.value = e;
        o.textContent = e.charAt(0).toUpperCase() + e.slice(1);
        if (e === options.defaults.tts_engine) o.selected = true;
        ttsSel.appendChild(o);
      });

      updateVoices();
      ttsSel.addEventListener("change", updateVoices);
    }

    function updateVoices() {
      const engine = document.getElementById("tts_engine").value;
      const voices = engine === "kokoro" ? options.kokoro_voices : options.vibevoice_voices;
      const voiceSel = document.getElementById("voice");
      voiceSel.innerHTML = "";
      voices.forEach(v => {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = v;
        voiceSel.appendChild(o);
      });
      document.getElementById("voice-row").style.display = voices.length > 1 ? "block" : "none";
    }

    document.getElementById("submit").addEventListener("click", async () => {
      const topic = document.getElementById("topic").value.trim();
      if (!topic) { alert("Please enter a topic."); return; }

      document.getElementById("submit").disabled = true;
      document.getElementById("error-text").textContent = "";
      document.getElementById("video-section").style.display = "none";
      document.getElementById("status-card").style.display = "block";
      setProgress("Starting…", 5);

      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic,
          quality:      document.getElementById("quality").value,
          level:        document.getElementById("level").value,
          duration:     parseInt(document.getElementById("duration").value),
          voice:        document.getElementById("voice").value,
          tts_engine:   document.getElementById("tts_engine").value,
          llm_provider: document.getElementById("llm_provider").value,
        }),
      });

      const { job_id } = await res.json();
      pollInterval = setInterval(() => poll(job_id), 2000);
    });

    async function poll(job_id) {
      const res = await fetch(`/api/status/${job_id}`);
      const job = await res.json();

      setProgress(job.step || "Working…", job.pct || 0);

      if (job.status === "complete") {
        clearInterval(pollInterval);
        setProgress("Done!", 100);
        const videoUrl = `/api/download/${job_id}`;
        const player = document.getElementById("video-player");
        player.src = videoUrl;
        document.getElementById("download-btn").href = videoUrl;
        document.getElementById("download-btn").download = `mathmotion_${job_id}.mp4`;
        document.getElementById("video-section").style.display = "block";
        document.getElementById("submit").disabled = false;
      } else if (job.status === "failed") {
        clearInterval(pollInterval);
        document.getElementById("error-text").textContent = "Error: " + job.error;
        setProgress("Failed", 0);
        document.getElementById("submit").disabled = false;
      }
    }

    function setProgress(step, pct) {
      document.getElementById("step-text").textContent = step;
      document.getElementById("progress-bar").style.width = pct + "%";
    }

    loadOptions();
  </script>
</body>
</html>
```

**Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: web UI with topic input, provider/voice/quality selection, progress bar, and video player"
```

---

## Task 14: Smoke Test

**Step 1: Set up environment**

```bash
cp .env.example .env
# Edit .env — add GEMINI_API_KEY at minimum
```

**Step 2: Install system dependencies (if not already installed)**

```bash
# Ubuntu/Debian
sudo apt install ffmpeg texlive-latex-base texlive-fonts-recommended \
                 texlive-latex-extra libcairo2-dev libpango1.0-dev
```

**Step 3: Start the app**

```bash
uv run python app.py
```
Expected: Browser opens at `http://127.0.0.1:8000`

**Step 4: Test via UI**

1. Type: "Explain what a derivative is and why it matters geometrically"
2. Set quality to **Draft**
3. Set duration to **60s**
4. Click **Generate Video**
5. Wait for progress bar to reach 100%
6. Video should appear in the browser and be playable

**Step 5: Run unit tests**

```bash
uv run pytest tests/ -v
```
Expected: All PASS

**Step 6: Update Claude.md progress and commit**

```bash
git add .
git commit -m "feat: complete personal-use web app — smoke test passing"
```

---

## Quick Reference

```bash
# Start the app
uv run python app.py
# Opens browser at http://127.0.0.1:8000

# Run tests
uv run pytest tests/ -v

# Change LLM provider: edit config.yaml → llm.provider
# Change TTS engine: edit config.yaml → tts.engine
# Change quality default: edit config.yaml → manim.default_quality
```

## System Requirements

- Python 3.13+
- `ffmpeg` + `ffprobe`
- LaTeX + Cairo (for Manim)
- `GEMINI_API_KEY` in `.env` (or `OPENROUTER_API_KEY` if using OpenRouter)

```bash
# Ubuntu/Debian system deps
sudo apt install ffmpeg texlive-latex-base texlive-fonts-recommended \
                 texlive-latex-extra libcairo2-dev libpango1.0-dev
```
