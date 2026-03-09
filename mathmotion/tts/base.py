from abc import ABC, abstractmethod
from pathlib import Path


class TTSEngine(ABC):
    @abstractmethod
    def synthesise(self, text: str, output_path: Path, voice: str = None, speed: float = 1.0) -> float:
        """Synthesise text → WAV at output_path. Returns actual duration in seconds."""
        ...

    @abstractmethod
    def available_voices(self) -> list[str]: ...
