# MathMotion

Turns a plain-English math topic into a narrated, animated video — locally, in your browser.

Type a topic like *"Explain what a derivative is"* and get an `.mp4` with Manim animations and a spoken narration, generated end-to-end by an LLM.

---

## Example output

<video src="assets/example.mp4" controls width="100%"></video>

---

## How it works

```
Your topic
   │
   ▼
LLM generates Manim animation code + narration script
   │
   ├─────────────────────┐
   ▼                     ▼
Manim renders video   Kokoro/Vibevoice synthesises audio   ← runs in parallel
   │                     │
   └──────────┬──────────┘
              ▼
         FFmpeg combines into final.mp4
              │
              ▼
         Plays in your browser
```

---

## Requirements

**Python:** 3.13+

**System packages:**

```bash
# Ubuntu / Debian
sudo apt install \
  ffmpeg \
  libcairo2-dev libpango1.0-dev \
  texlive-latex-base texlive-fonts-recommended texlive-latex-extra
```

**API key** (at least one):

| Provider | Key | Notes |
|---|---|---|
| Gemini (default) | `GEMINI_API_KEY` | Free tier available at [aistudio.google.com](https://aistudio.google.com) |
| OpenRouter | `OPENROUTER_API_KEY` | Optional alternative |
| Ollama | — | Free, runs locally — install [Ollama](https://ollama.com) separately |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/harikc456/3b1b-at-home.git
cd 3b1b-at-home

# 2. Install Python dependencies
uv sync

# 3. Add your API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=AIza...

# 4. Start
uv run python app.py
```

The browser opens automatically at `http://127.0.0.1:8000`.

---

## Using the web UI

1. **Type a topic** — anything mathematical, e.g.:
   - *"Explain the geometric intuition behind eigenvalues"*
   - *"What is the Fourier transform and why does it matter?"*
   - *"Prove that the square root of 2 is irrational"*

2. **Choose settings:**

   | Setting | Options | Notes |
   |---|---|---|
   | Level | High School / Undergraduate / Graduate | Adjusts explanation depth |
   | Duration | 1 – 5 minutes | Target length |
   | Quality | Draft / Standard / High | Draft is fastest for testing |
   | AI Model | Gemini / OpenRouter / Ollama | Must have the key configured |
   | Voice Engine | Kokoro / Vibevoice | Kokoro has better quality |
   | Voice | Various | Depends on chosen engine |

3. **Click Generate Video** and wait. Progress is shown live.

4. The video plays inline when done. Use the **Download** button to save it.

---

## Configuration

All settings live in `config.yaml`. Key things you might want to change:

```yaml
llm:
  provider: gemini          # Switch to openrouter or ollama

tts:
  engine: kokoro            # Switch to vibevoice

manim:
  default_quality: standard # draft | standard | high
  background_color: "#1a1a2e"

composition:
  sync_strategy: time_stretch  # or audio_led (higher quality, slower)

server:
  host: 127.0.0.1
  port: 8000
```

### Switching to Ollama (fully local, no API key)

1. Install Ollama: [ollama.com](https://ollama.com)
2. Pull a model: `ollama pull gemma3:27b`
3. Edit `config.yaml`: set `llm.provider: ollama`

---

## Project structure

```
├── app.py                  # Start here — FastAPI server
├── config.yaml             # All settings
├── .env                    # Your API keys (not committed)
├── static/
│   └── index.html          # Web UI
├── api/
│   └── routes.py           # API endpoints
├── mathmotion/
│   ├── pipeline.py         # Orchestrates all stages
│   ├── llm/                # Gemini, OpenRouter, Ollama
│   ├── tts/                # Kokoro, Vibevoice
│   ├── stages/
│   │   ├── generate.py     # LLM → Manim code + narration
│   │   ├── render.py       # Manim → video files
│   │   ├── tts.py          # Text → audio files
│   │   └── compose.py      # Video + audio → final.mp4
│   ├── schemas/            # Pydantic models
│   └── utils/              # Config, validation, ffprobe
├── prompts/
│   ├── system_prompt.txt   # LLM instruction template
│   └── domain_hints/       # Extra context per math domain
├── jobs/                   # Generated videos (gitignored)
└── tests/                  # Unit tests
```

---

## Running tests

```bash
uv run --no-sync pytest tests/ -v
```

---

## Troubleshooting

**`pycairo` build fails**
→ Install `libcairo2-dev` (see Requirements above)

**Manim render fails**
→ Install the LaTeX packages (`texlive-*`) listed in Requirements

**Video has no audio / silence**
→ Kokoro model downloads on first use — wait for it to finish, then retry

**Ollama times out**
→ Increase `llm.timeout_seconds` in `config.yaml` (local inference is slow for large models)

**`GEMINI_API_KEY` not found**
→ Make sure `.env` exists and contains the key — copy from `.env.example`
