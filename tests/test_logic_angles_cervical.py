import math
import pytest
import logic_angles_cervical as cerv


# ---------------------------------------------------------------------------
# c2c7_angle_deg
# ---------------------------------------------------------------------------

def test_c2c7_angle_both_horizontal():
    """Both endplates horizontal → C2C7 = 0°."""
    assert abs(cerv.c2c7_angle_deg((0, 0), (1, 0), (0, 5), (1, 5)) - 0.0) < 1e-9


def test_c2c7_angle_tilted_c7_only():
    """C2 horizontal, C7 tilted → angle equals C7 slope difference."""
    # C7 tilted so that slope is 45°  →  slope(C7) - slope(C2) = 45 - 0 = 45
    c7_ant = (0.0, 5.0)
    c7_post = (1.0, 4.0)  # rises 1 to the right, slope = +45° (y-down coords) → signed_slope = -45° → wait
    # In y-down image coords: vec (1,-1) → atan2(-1,1) = -45° → -(-45) = 45
    # C2: vec (1,0) → slope 0
    # LL formula: slope(C7) - slope(C2) = 45 - 0 = 45
    result = cerv.c2c7_angle_deg((0, 0), (1, 0), c7_ant, c7_post)
    assert abs(result - 45.0) < 1e-6


def test_c2c7_angle_matches_lumbosacral_formula():
    """Verify c2c7_angle_deg uses the same formula as lumbosacral_lordosis_deg."""
    import sys, os
    lumbar_lib = os.path.join(os.path.dirname(__file__), "..", "LumbarMeasureAssist", "lib")
    if lumbar_lib not in sys.path:
        sys.path.insert(0, lumbar_lib)
    from logic_angles import lumbosacral_lordosis_deg, vector_from_points

    C2_ant, C2_post   = (10.0, 50.0), (60.0, 55.0)
    C7_ant, C7_post   = (15.0, 200.0), (65.0, 210.0)

    ref = lumbosacral_lordosis_deg(
        vector_from_points(C2_ant, C2_post),
        vector_from_points(C7_ant, C7_post),
    )
    ours = cerv.c2c7_angle_deg(C2_ant, C2_post, C7_ant, C7_post)
    assert math.isclose(ref, ours, abs_tol=1e-9)


def test_c2c7_angle_wrapped_in_range():
    result = cerv.c2c7_angle_deg((0, 0), (1, 0), (0, 5), (1, 5))
    assert -180.0 <= result <= 180.0


# ---------------------------------------------------------------------------
# t1_slope_deg
# ---------------------------------------------------------------------------

def test_t1_slope_horizontal():
    assert abs(cerv.t1_slope_deg((0, 0), (1, 0)) - 0.0) < 1e-9


def test_t1_slope_known_angle():
    # vec (1, -1) in y-down coords → atan2(-1,1) = -45° → -(-45) = 45°
    assert abs(cerv.t1_slope_deg((0, 1), (1, 0)) - 45.0) < 1e-6


def test_t1_slope_zero_vector_raises():
    with pytest.raises(ValueError):
        cerv.t1_slope_deg((1.0, 1.0), (1.0, 1.0))


# ---------------------------------------------------------------------------
# c2c7_sva_mm
# ---------------------------------------------------------------------------

def test_c2c7_sva_zero():
    assert cerv.c2c7_sva_mm((5.0, 0.0), (5.0, 0.0)) == 0.0


def test_c2c7_sva_positive_when_c2_anterior():
    """C2_center anterior (larger x) → positive SVA."""
    assert cerv.c2c7_sva_mm((10.0, 0.0), (5.0, 0.0)) > 0.0


def test_c2c7_sva_scale_by_spacing():
    """10 px offset * 0.3 mm/px = 3.0 mm."""
    result = cerv.c2c7_sva_mm((10.0, 0.0), (0.0, 0.0), pixel_spacing_mm=0.3)
    assert math.isclose(result, 3.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_cervical_measurements
# ---------------------------------------------------------------------------

def _make_valid_points():
    return {
        "C2_ant":      (0.0, 0.0),
        "C2_post":     (1.0, 0.0),
        "C2_center":   (0.5, 0.0),
        "C7_inf_ant":  (0.0, 5.0),
        "C7_inf_post": (1.0, 5.0),
        "C7_sup_post": (1.0, 4.5),
        "T1_ant":      (0.0, 6.0),
        "T1_post":     (1.0, 6.0),
    }


def test_compute_cervical_returns_three_keys():
    result = cerv.compute_cervical_measurements(_make_valid_points())
    assert set(result.keys()) == {"C2C7_angle", "T1S", "C2C7_SVA"}


def test_compute_cervical_missing_key_raises():
    pts = _make_valid_points()
    del pts["T1_ant"]
    with pytest.raises(ValueError, match="Missing points"):
        cerv.compute_cervical_measurements(pts)


def test_compute_cervical_all_horizontal_gives_zero_angles():
    pts = _make_valid_points()
    result = cerv.compute_cervical_measurements(pts)
    assert abs(result["C2C7_angle"] - 0.0) < 1e-9
    assert abs(result["T1S"] - 0.0) < 1e-9


def test_compute_cervical_sva_uses_spacing():
    pts = _make_valid_points()
    pts["C2_center"]   = (11.0, 0.0)
    pts["C7_sup_post"] = (1.0, 0.0)
    r1 = cerv.compute_cervical_measurements(pts, pixel_spacing_mm=1.0)
    r2 = cerv.compute_cervical_measurements(pts, pixel_spacing_mm=0.5)
    assert math.isclose(r1["C2C7_SVA"], 2 * r2["C2C7_SVA"], abs_tol=1e-9)
