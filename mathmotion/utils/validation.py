import ast

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
