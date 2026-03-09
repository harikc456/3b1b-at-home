import logging
import re
from pathlib import Path

from mathmotion.llm.base import LLMProvider
from mathmotion.utils.errors import LLMError, ValidationError
from mathmotion.utils.validation import check_forbidden_calls, check_forbidden_imports

logger = logging.getLogger(__name__)

REPAIR_SYSTEM_PROMPT = """\
You are an expert Manim animator. You will be given a broken Manim Python scene and the error it produced. Fix the code.

Rules:
- Preserve the exact class name
- Preserve all # WAIT:{seg_id} comments and the self.wait() lines that follow them
- Only import what you use (never use `from manim import *`)
- Forbidden imports: os, sys, subprocess, socket, urllib, requests, httpx
- Forbidden calls: open(), exec(), eval(), __import__()
- Return ONLY the raw Python code — no markdown fences, no explanation, nothing else
"""


def _strip_fences(code: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` wrappers if present."""
    code = code.strip()
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


def _validate_code(code: str) -> None:
    """Raises ValidationError if code has syntax errors or forbidden content."""
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        raise ValidationError(f"Fixed code has syntax error: {e}")

    bad = check_forbidden_imports(code)
    if bad:
        raise ValidationError(f"Fixed code has forbidden imports: {bad}")

    bad = check_forbidden_calls(code)
    if bad:
        raise ValidationError(f"Fixed code has forbidden calls: {bad}")


def fix_scene(scene_file: Path, stderr: str, provider: LLMProvider) -> str:
    """
    Ask the LLM to fix a broken Manim scene file.

    Reads scene_file, sends code + stderr to the LLM, validates the response,
    overwrites scene_file with the fixed code, and returns the fixed code.

    Raises ValidationError if the LLM's fix is itself invalid (scene_file is
    NOT overwritten in that case).
    """
    original_code = scene_file.read_text()
    user_prompt = (
        f"The following Manim scene failed to render.\n\n"
        f"ERROR:\n{stderr}\n\n"
        f"CODE:\n{original_code}\n\n"
        f"Fix the code."
    )

    logger.info(f"Requesting LLM repair for {scene_file.name}")
    try:
        resp = provider.complete(
            REPAIR_SYSTEM_PROMPT,
            user_prompt,
            json_mode=False,
        )
    except LLMError as e:
        raise ValidationError(f"LLM repair call failed: {e}") from e

    fixed_code = _strip_fences(resp.content)
    _validate_code(fixed_code)

    scene_file.write_text(fixed_code)
    logger.info(f"Repair written to {scene_file.name}")
    return fixed_code
