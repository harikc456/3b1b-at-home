## 3b1b-at-home

3b1b-at-home is a project inspired by the Youtube channel 3Blue1Brown which makes educational mathematical videos with visualization to help the viewers understand complex concepts. The goal of the project is to make such educational videos of user's interest to help the user grasp otherwise difficult mathematical topics.

## Development Rules

- Package Management

    - ONLY use uv, NEVER pip
    - Installation: uv add package
    - Running tools: uv run tool
    - Upgrading: uv add --dev package --upgrade-package package
    - FORBIDDEN: uv pip install, @latest syntax

- Use .env file to store and load sensitive information like API keys
- Prefer explicit, readable code over clever one-liners
- Keep functions focused — one responsibility per function
- ASK questions and resolve any ambguity before starting the coding task
- Update each the Current status under the Development Roadmap after finishing the task
- Refer [spec.md](./spec.md) for techincal specifications and roadmap
- Update the progress section after major milestones, to keep track of the progress of this project
## Self-Improvement Loop

- After **any** correction from the user: record the pattern in `lessons.md`
- Write a concrete rule to prevent the same mistake recurring
- Review `lessons.md` at the start of every session before writing any code


## Development Roadmap

Full plan: [`docs/plans/2026-03-08-mathmotion-pipeline.md`](./docs/plans/2026-03-08-mathmotion-pipeline.md)

Personal-use web app: FastAPI backend + single HTML page. No auth, no Docker, no job queue.
LLM: Gemini (primary), OpenRouter, Ollama. TTS: Kokoro, Vibevoice.

- [x] Task 1: Project scaffold + dependencies
- [x] Task 2: Config system + errors (`mathmotion/utils/config.py`, `errors.py`)
- [x] Task 3: Pydantic schema (`mathmotion/schemas/script.py`)
- [x] Task 4: LLM providers — Gemini, OpenRouter, Ollama (`mathmotion/llm/`)
- [x] Task 5: TTS engines — Kokoro, Vibevoice (`mathmotion/tts/`)
- [x] Task 6: Validation + ffprobe utils (`mathmotion/utils/`)
- [x] Task 7: Stage 2 — LLM Script Generation (`mathmotion/stages/generate.py`)
- [x] Task 8: Stage 3 — Manim Render (`mathmotion/stages/render.py`)
- [x] Task 9: Stage 4 — TTS Synthesis (`mathmotion/stages/tts.py`)
- [x] Task 10: Stage 5 — A/V Composition (`mathmotion/stages/compose.py`)
- [x] Task 11: Pipeline orchestrator (`mathmotion/pipeline.py`)
- [x] Task 12: FastAPI backend (`app.py`, `api/routes.py`)
- [x] Task 13: Web UI (`static/index.html`)
- [ ] Task 14: Smoke test — pending system deps (libcairo2-dev, ffmpeg, texlive)

## Progress

**2026-03-08** — All 13 implementation tasks complete. 12/12 unit tests passing.
Pending: install system deps (libcairo2-dev, ffmpeg, texlive) then run smoke test.
Start app with: `uv run python app.py`