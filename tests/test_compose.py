from mathmotion.stages.compose import compute_drift, inject_actual_durations


def test_drift_calculation():
    assert abs(compute_drift(3.5, 3.0) - 0.1667) < 0.001


def test_drift_zero_estimated():
    assert compute_drift(3.5, 0.0) == 0.0


def test_inject_replaces_wait():
    code = "# WAIT:seg_001_a\nself.wait(3.8)"
    result = inject_actual_durations(code, {"seg_001_a": 4.2})
    assert "self.wait(4.200)" in result


def test_inject_ignores_unknown_segment():
    code = "# WAIT:seg_999\nself.wait(3.0)"
    result = inject_actual_durations(code, {"seg_001_a": 4.2})
    assert "self.wait(3.0)" in result


def test_inject_leaves_unrelated_lines():
    code = "x = 1\n# WAIT:seg_001_a\nself.wait(3.0)\ny = 2"
    result = inject_actual_durations(code, {"seg_001_a": 5.0})
    assert "x = 1" in result
    assert "y = 2" in result
    assert "self.wait(5.000)" in result
