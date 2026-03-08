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
