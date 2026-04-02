import ast

from mathmotion.schemas.script import GeneratedScript
from mathmotion.utils.errors import ValidationError

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "urllib", "urllib3",
    "requests", "httpx", "aiohttp", "shutil",
}
FORBIDDEN_CALLS = {"eval", "exec", "open", "__import__"}


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


def check_forbidden_calls(code: str) -> list[str]:
    """Returns list of forbidden call names found. Empty list = pass."""
    tree = ast.parse(code)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                found.append(node.func.id)
    return found


def validate_scene_item(scene) -> None:
    """Validate a single Scene. Raises ValidationError if any check fails."""
    try:
        ast.parse(scene.manim_code)
    except SyntaxError as e:
        raise ValidationError(f"Syntax error: {e}")

    bad = check_forbidden_imports(scene.manim_code)
    if bad:
        raise ValidationError(f"Forbidden imports: {bad}")

    bad = check_forbidden_calls(scene.manim_code)
    if bad:
        raise ValidationError(f"Forbidden calls: {bad}")

    if "MathMotionScene" not in scene.manim_code:
        raise ValidationError(
            f"Scene must extend MathMotionScene — got code without 'MathMotionScene'"
        )

    if scene.class_name not in scene.manim_code:
        raise ValidationError(
            f"class_name '{scene.class_name}' not found in manim_code"
        )

    if not scene.narration_segments:
        raise ValidationError("Scene has no narration segments — add self.voiceover() calls")

    if "self.voiceover(" not in scene.manim_code:
        raise ValidationError("No self.voiceover() calls found in scene code")

    for seg in scene.narration_segments:
        if not seg.text.strip():
            raise ValidationError(f"Narration segment {seg.id} has empty text")


def validate_script(data: dict) -> GeneratedScript:
    """Validate a GeneratedScript dict. Raises ValidationError with a message suitable for retry."""
    try:
        script = GeneratedScript.model_validate(data)
    except Exception as e:
        raise ValidationError(f"Schema error: {e}")

    for scene in script.scenes:
        validate_scene_item(scene)

    return script
