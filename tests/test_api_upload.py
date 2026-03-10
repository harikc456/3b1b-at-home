# tests/test_api_upload.py
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

MINIMAL_SCENE_CODE = """from manim import *

class SceneA(Scene):
    def construct(self):
        self.wait(1)
"""

VALID_SCRIPT = {
    "title": "Test",
    "topic": "derivatives",
    "scenes": [
        {
            "id": "scene_a",
            "class_name": "SceneA",
            "manim_code": MINIMAL_SCENE_CODE,
            "narration_segments": [
                {"id": "seg_1", "text": "Hello world", "cue_offset": 0.0}
            ],
        }
    ],
}


def test_upload_valid_script_returns_job_id():
    with patch("api.routes.threading.Thread") as mock_thread:
        mock_thread.return_value.start = lambda: None
        resp = client.post(
            "/api/generate-from-script",
            files={"file": ("script.json", json.dumps(VALID_SCRIPT), "application/json")},
            data={"quality": "draft"},
        )
    assert resp.status_code == 200
    assert "job_id" in resp.json()


def test_upload_invalid_json_returns_error():
    resp = client.post(
        "/api/generate-from-script",
        files={"file": ("script.json", b"not valid json", "application/json")},
    )
    assert resp.status_code == 400
    assert "Invalid JSON" in resp.json()["detail"]


def test_upload_invalid_schema_returns_error():
    bad_script = {"title": "X"}  # missing required fields
    resp = client.post(
        "/api/generate-from-script",
        files={"file": ("script.json", json.dumps(bad_script), "application/json")},
    )
    assert resp.status_code == 400


def test_upload_forbidden_import_returns_error():
    forbidden_script = {
        "title": "Test",
        "topic": "derivatives",
        "scenes": [
            {
                "id": "scene_a",
                "class_name": "SceneA",
                "manim_code": "import os\nfrom manim import *\nclass SceneA(Scene):\n    def construct(self): pass",
                "narration_segments": [
                    {"id": "seg_1", "text": "Hello world", "cue_offset": 0.0}
                ],
            }
        ],
    }
    resp = client.post(
        "/api/generate-from-script",
        files={"file": ("script.json", json.dumps(forbidden_script), "application/json")},
    )
    assert resp.status_code == 400
    assert "forbidden" in resp.json()["detail"].lower()
