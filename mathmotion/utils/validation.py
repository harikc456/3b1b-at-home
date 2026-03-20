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


def validate_script(data: dict) -> GeneratedScript:
    """Validate a GeneratedScript dict. Raises ValidationError with a message suitable for retry."""
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
            raise ValidationError(f"Scene {scene.id} has forbidden imports: {bad}")

        bad = check_forbidden_calls(scene.manim_code)
        if bad:
            raise ValidationError(f"Scene {scene.id} has forbidden calls: {bad}")

        if scene.class_name not in scene.manim_code:
            raise ValidationError(
                f"Scene {scene.id}: class_name '{scene.class_name}' not found in manim_code"
            )

        for seg in scene.narration_segments:
            if not seg.text.strip():
                raise ValidationError(f"Narration segment {seg.id} has empty text")

    return script
