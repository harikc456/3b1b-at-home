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


def test_validate_scene_item_passes_for_mathmotion_scene():
    from mathmotion.utils.validation import validate_scene_item
    from mathmotion.schemas.script import Scene, NarrationSegment
    code = (
        "from manim import *\n"
        "from mathmotion.manim_ext import MathMotionScene\n"
        "class Scene_Test(MathMotionScene):\n"
        "    def construct(self):\n"
        "        with self.voiceover('hello world') as tracker:\n"
        "            self.play(Write(Text('hi')), run_time=tracker.duration)\n"
    )
    scene = Scene(
        id="s1", class_name="Scene_Test", manim_code=code,
        narration_segments=[NarrationSegment(id="seg_0", text="hello world")],
    )
    validate_scene_item(scene)  # must not raise


def test_validate_scene_item_rejects_missing_mathmotion_scene():
    from mathmotion.utils.validation import validate_scene_item
    from mathmotion.schemas.script import Scene, NarrationSegment
    from mathmotion.utils.errors import ValidationError
    code = (
        "from manim import *\n"
        "class Scene_Test(Scene):\n"
        "    def construct(self): pass\n"
    )
    scene = Scene(
        id="s1", class_name="Scene_Test", manim_code=code,
        narration_segments=[NarrationSegment(id="seg_0", text="hello")],
    )
    with pytest.raises(ValidationError, match="MathMotionScene"):
        validate_scene_item(scene)


def test_validate_scene_item_rejects_no_voiceover_calls():
    from mathmotion.utils.validation import validate_scene_item
    from mathmotion.schemas.script import Scene, NarrationSegment
    from mathmotion.utils.errors import ValidationError
    code = (
        "from manim import *\n"
        "from mathmotion.manim_ext import MathMotionScene\n"
        "class Scene_Test(MathMotionScene):\n"
        "    def construct(self): self.wait(1)\n"
    )
    scene = Scene(
        id="s1", class_name="Scene_Test", manim_code=code,
        narration_segments=[NarrationSegment(id="seg_0", text="hello")],
    )
    with pytest.raises(ValidationError, match="voiceover"):
        validate_scene_item(scene)


def test_validate_scene_item_rejects_empty_segment_text():
    from mathmotion.utils.validation import validate_scene_item
    from mathmotion.schemas.script import Scene, NarrationSegment
    from mathmotion.utils.errors import ValidationError
    code = (
        "from manim import *\n"
        "from mathmotion.manim_ext import MathMotionScene\n"
        "class Scene_Test(MathMotionScene):\n"
        "    def construct(self):\n"
        "        with self.voiceover('') as t: self.wait(t.duration)\n"
    )
    scene = Scene(
        id="s1", class_name="Scene_Test", manim_code=code,
        narration_segments=[NarrationSegment(id="seg_0", text="  ")],
    )
    with pytest.raises(ValidationError):
        validate_scene_item(scene)
