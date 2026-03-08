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
                raise ConfigError(f"Missing environment variable: {m.group(1)}")
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
        raise ConfigError(f"Config file not found: {config_path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(_resolve_env(raw))
