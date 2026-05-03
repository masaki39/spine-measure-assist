import pytest

from measurement_sets import PELVIC_SET, MeasurementSetDef, get_set, set_names


def test_set_names_contains_pelvic():
    assert "Pelvic Parameters" in set_names()


def test_set_names_returns_list():
    assert isinstance(set_names(), list)


def test_get_set_returns_correct_object():
    assert get_set("Pelvic Parameters") is PELVIC_SET


def test_get_set_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_set("NonExistentSet")


# --- MeasurementSetDef type ---

def test_pelvic_set_is_measurement_set_def():
    assert isinstance(PELVIC_SET, MeasurementSetDef)


# --- point_labels ---

def test_pelvic_set_has_six_labels():
    assert len(PELVIC_SET.point_labels) == 6


def test_pelvic_set_contains_required_labels():
    labels = set(PELVIC_SET.point_labels)
    for expected in ["L1_ant", "L1_post", "S1_ant", "S1_post", "FH", "L1_center"]:
        assert expected in labels, f"Missing label: {expected}"


def test_pelvic_set_labels_are_unique():
    assert len(PELVIC_SET.point_labels) == len(set(PELVIC_SET.point_labels))


# --- angle_names ---

def test_pelvic_set_has_five_angle_names():
    assert len(PELVIC_SET.angle_names) == 5


def test_pelvic_set_angle_names_correct():
    assert set(PELVIC_SET.angle_names) == {"PI", "PT", "SS", "LL", "L1PA"}


def test_pelvic_set_angle_names_are_unique():
    assert len(PELVIC_SET.angle_names) == len(set(PELVIC_SET.angle_names))


# --- vector_definitions ---

def test_pelvic_set_vector_definitions_has_four_entries():
    assert len(PELVIC_SET.vector_definitions) == 4


def test_pelvic_set_vector_definitions_keys():
    keys = set(PELVIC_SET.vector_definitions.keys())
    assert keys == {"L1", "S1", "pelvis", "L1_pelvis"}


def test_pelvic_set_l1_vector_endpoints():
    p1, p2 = PELVIC_SET.vector_definitions["L1"]
    assert p1 == "L1_ant"
    assert p2 == "L1_post"


def test_pelvic_set_s1_vector_endpoints():
    p1, p2 = PELVIC_SET.vector_definitions["S1"]
    assert p1 == "S1_ant"
    assert p2 == "S1_post"


def test_pelvic_set_vector_endpoints_are_valid():
    valid_keys = set(PELVIC_SET.point_labels) | set(PELVIC_SET.midpoint_definitions.keys())
    for name, (p1, p2) in PELVIC_SET.vector_definitions.items():
        assert p1 in valid_keys, f"{name}: endpoint '{p1}' not in labels or midpoints"
        assert p2 in valid_keys, f"{name}: endpoint '{p2}' not in labels or midpoints"


# --- vector_modes ---

def test_pelvic_set_vector_modes_match_definitions():
    assert set(PELVIC_SET.vector_modes.keys()) == set(PELVIC_SET.vector_definitions.keys())


def test_pelvic_set_vector_modes_valid_values():
    for name, mode in PELVIC_SET.vector_modes.items():
        assert mode in ("Line", "Segment"), f"{name} has invalid mode: {mode}"


def test_pelvic_set_l1_s1_are_line_mode():
    assert PELVIC_SET.vector_modes["L1"] == "Line"
    assert PELVIC_SET.vector_modes["S1"] == "Line"


def test_pelvic_set_pelvis_vectors_are_segment():
    assert PELVIC_SET.vector_modes["pelvis"] == "Segment"
    assert PELVIC_SET.vector_modes["L1_pelvis"] == "Segment"


# --- vector_colors ---

def test_pelvic_set_vector_colors_match_definitions():
    assert set(PELVIC_SET.vector_colors.keys()) == set(PELVIC_SET.vector_definitions.keys())


def test_pelvic_set_vector_colors_rgb_range():
    for name, (r, g, b) in PELVIC_SET.vector_colors.items():
        assert 0.0 <= r <= 1.0, f"{name}: red={r} out of [0,1]"
        assert 0.0 <= g <= 1.0, f"{name}: green={g} out of [0,1]"
        assert 0.0 <= b <= 1.0, f"{name}: blue={b} out of [0,1]"


def test_pelvic_set_vector_colors_are_tuples_of_three():
    for name, color in PELVIC_SET.vector_colors.items():
        assert len(color) == 3, f"{name} color should have 3 components"


# --- midpoint_definitions ---

def test_pelvic_set_has_s1_mid():
    assert "_S1_mid" in PELVIC_SET.midpoint_definitions


def test_pelvic_set_s1_mid_endpoints():
    a, b = PELVIC_SET.midpoint_definitions["_S1_mid"]
    assert a == "S1_ant"
    assert b == "S1_post"


def test_pelvic_set_midpoint_keys_in_valid_labels():
    label_set = set(PELVIC_SET.point_labels)
    for mid_name, (a, b) in PELVIC_SET.midpoint_definitions.items():
        assert a in label_set, f"midpoint {mid_name}: '{a}' not in labels"
        assert b in label_set, f"midpoint {mid_name}: '{b}' not in labels"


# --- compute_fn ---

def test_pelvic_set_compute_fn_is_callable():
    assert callable(PELVIC_SET.compute_fn)


def test_pelvic_set_compute_fn_returns_dict():
    pts = {
        "L1_ant": (-0.5, 5.0), "L1_post": (0.5, 5.0),
        "S1_ant": (-0.5, 0.0), "S1_post": (0.5, 0.0),
        "FH": (0.0, 10.0),
    }
    result = PELVIC_SET.compute_fn(pts)
    assert isinstance(result, dict)


def test_pelvic_set_compute_fn_returns_base_angles():
    pts = {
        "L1_ant": (-0.5, 5.0), "L1_post": (0.5, 5.0),
        "S1_ant": (-0.5, 0.0), "S1_post": (0.5, 0.0),
        "FH": (0.0, 10.0),
    }
    result = PELVIC_SET.compute_fn(pts)
    for angle in ["PI", "PT", "SS", "LL"]:
        assert angle in result


def test_pelvic_set_compute_fn_includes_l1pa_with_l1_center():
    pts = {
        "L1_ant": (-0.5, 5.0), "L1_post": (0.5, 5.0),
        "S1_ant": (-0.5, 0.0), "S1_post": (0.5, 0.0),
        "FH": (0.0, 10.0), "L1_center": (1.0, 5.0),
    }
    result = PELVIC_SET.compute_fn(pts)
    assert "L1PA" in result
