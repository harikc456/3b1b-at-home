import pytest
from mathmotion.utils.validation import check_forbidden_imports, check_forbidden_calls


def test_detects_subprocess():
    assert "subprocess" in check_forbidden_imports("import subprocess")


def test_detects_os():
    assert "os" in check_forbidden_imports("import os\nprint(os.getcwd())")


def test_clean_imports_pass():
    assert check_forbidden_imports("from manim import *\nimport math") == []


def test_detects_eval():
    assert "eval" in check_forbidden_calls("x = eval('1+1')")


def test_detects_exec():
    assert "exec" in check_forbidden_calls("exec('import os')")


def test_detects_open():
    assert "open" in check_forbidden_calls("f = open('file.txt')")


def test_clean_calls_pass():
    code = "from manim import *\nclass Scene_Test(Scene):\n    def construct(self): pass"
    assert check_forbidden_calls(code) == []


def test_validate_script_importable_from_validation():
    from mathmotion.utils.validation import validate_script
    assert callable(validate_script)


def test_validate_script_rejects_invalid_schema():
    from mathmotion.utils.validation import validate_script
    from mathmotion.utils.errors import ValidationError
    with pytest.raises(ValidationError):
        validate_script({"title": "X"})  # missing required fields
