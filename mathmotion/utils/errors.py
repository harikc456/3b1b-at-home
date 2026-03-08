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
