import math
import pytest

import SagittalMeasureAssist.lib.logic_angles as angles


def test_vector_from_points():
    assert angles.vector_from_points((0, 0), (1, 1)) == (1, 1)


def test_vector_length_zero():
    assert angles.vector_length((0, 0)) == 0


@pytest.mark.parametrize(
    "v,expected",
    [
        ((1, 0), 0.0),
        ((0, 1), -90.0),  # y増加が下向きなので負
        ((0, -1), 90.0),
        ((-1, 0), 0.0),
    ],
)
def test_signed_slope_angle_deg(v, expected):
    assert math.isclose(angles.signed_slope_angle_deg(v), expected, abs_tol=1e-6)


@pytest.mark.parametrize(
    "v,expected",
    [
        ((0, -1), 0.0),
        ((1, 0), 90.0),
        ((-1, 0), -90.0),
        ((0, 1), 180.0 - 180.0),  # wrapped to 0 after branch
    ],
)
def test_signed_vertical_angle_deg(v, expected):
    assert math.isclose(angles.signed_vertical_angle_deg(v), expected, abs_tol=1e-6)


def test_pelvic_incidence_deg_basic():
    v_pelvis = (0, -1)
    v_S1 = (1, 0)
    assert math.isclose(angles.pelvic_incidence_deg(v_pelvis, v_S1), 0.0, abs_tol=1e-6)


def test_compute_angles_from_points():
    pts = {
        "FH": (0.5, 2.0),
        "S1_ant": (0.0, 0.0),
        "S1_post": (1.0, 1.0),
        "L1_ant": (0.0, 1.0),
        "L1_post": (1.0, 2.0),
    }
    result = angles.compute_angles_from_points(pts)
    assert set(result.keys()) == {"PI", "PT", "SS", "LL"}
    assert math.isclose(result["SS"], -45.0, abs_tol=0.5)
    assert math.isclose(result["LL"], 0.0, abs_tol=0.5)
    assert math.isclose(result["PT"], 0.0, abs_tol=0.5)
    assert math.isclose(result["PI"], 45.0, abs_tol=0.5)


def test_compute_angles_missing():
    with pytest.raises(ValueError):
        angles.compute_angles_from_points({"FH": (0, 0)})


def test_normalize_basic():
    nx, ny = angles.normalize((3, 4))
    assert math.isclose(nx, 0.6, abs_tol=1e-6)
    assert math.isclose(ny, 0.8, abs_tol=1e-6)


def test_normalize_zero_raises():
    with pytest.raises(ValueError):
        angles.normalize((0, 0))


@pytest.mark.parametrize(
    "v1,v2,expected",
    [
        ((1, 0), (0, 1), 90.0),
        ((1, 0), (2, 0), 0.0),
        ((1, 0), (-1, 0), 180.0),
    ],
)
def test_angle_between_vectors(v1, v2, expected):
    assert math.isclose(angles.angle_between_vectors(v1, v2), expected, abs_tol=1e-6)


def test_angle_between_vectors_zero_raises():
    with pytest.raises(ValueError):
        angles.angle_between_vectors((0, 0), (1, 0))


@pytest.mark.parametrize(
    "angle,expected",
    [
        (90.0, 90.0),
        (270.0, -90.0),
        (-270.0, 90.0),
        (180.0, 180.0),
        (-180.0, -180.0),
    ],
)
def test_wrap_signed_angle(angle, expected):
    assert math.isclose(angles.wrap_signed_angle(angle), expected, abs_tol=1e-6)


def test_l1_pelvic_angle_deg_xplus_negative():
    # x+ side gives negative angle (sign convention of the implementation)
    FH = (0, 10)
    S1_mid = (0, 0)
    assert angles.l1_pelvic_angle_deg(FH, S1_mid, (1, 5)) < 0


def test_l1_pelvic_angle_deg_xminus_positive():
    # x- side gives positive angle
    FH = (0, 10)
    S1_mid = (0, 0)
    assert angles.l1_pelvic_angle_deg(FH, S1_mid, (-1, 5)) > 0


def test_compute_angles_includes_l1pa_when_l1_center_present():
    pts = {
        "FH": (0, 10), "S1_ant": (-0.5, 0), "S1_post": (0.5, 0),
        "L1_ant": (-0.5, 5), "L1_post": (0.5, 5), "L1_center": (1.0, 5),
    }
    result = angles.compute_angles_from_points(pts)
    assert "L1PA" in result
    assert result["L1PA"] < 0  # x+ side → negative per sign convention


def test_compute_angles_l1pa_positive_when_xminus():
    pts = {
        "FH": (0, 10), "S1_ant": (-0.5, 0), "S1_post": (0.5, 0),
        "L1_ant": (-0.5, 5), "L1_post": (0.5, 5), "L1_center": (-1.0, 5),
    }
    result = angles.compute_angles_from_points(pts)
    assert result["L1PA"] > 0  # x- side → positive per sign convention


def test_compute_angles_no_l1pa_without_l1_center():
    pts = {"FH": (0, 2), "S1_ant": (0, 0), "S1_post": (1, 1),
           "L1_ant": (0, 1), "L1_post": (1, 2)}
    result = angles.compute_angles_from_points(pts)
    assert "L1PA" not in result


def test_lumbosacral_lordosis_deg_same_slope():
    assert math.isclose(angles.lumbosacral_lordosis_deg((1, 0), (1, 0)), 0.0, abs_tol=1e-6)


def test_signed_slope_angle_deg_zero_raises():
    with pytest.raises(ValueError):
        angles.signed_slope_angle_deg((0, 0))


def test_signed_vertical_angle_deg_zero_raises():
    with pytest.raises(ValueError):
        angles.signed_vertical_angle_deg((0, 0))


# --- vector_from_points ---

@pytest.mark.parametrize("a,b,expected", [
    ((1, 2), (4, 6), (3, 4)),
    ((-1, -1), (1, 1), (2, 2)),
    ((5, 5), (5, 5), (0, 0)),
    ((0, 0), (-3, -4), (-3, -4)),
])
def test_vector_from_points_various(a, b, expected):
    assert angles.vector_from_points(a, b) == expected


# --- vector_length ---

@pytest.mark.parametrize("v,expected", [
    ((3, 4), 5.0),
    ((-3, -4), 5.0),
    ((1, 0), 1.0),
    ((0, 2), 2.0),
])
def test_vector_length_nonzero(v, expected):
    assert math.isclose(angles.vector_length(v), expected, abs_tol=1e-6)


# --- normalize ---

@pytest.mark.parametrize("v,ex,ey", [
    ((3, 0), 1.0, 0.0),
    ((0, 5), 0.0, 1.0),
    ((-4, 0), -1.0, 0.0),
    ((3, 4), 0.6, 0.8),
])
def test_normalize_values(v, ex, ey):
    nx, ny = angles.normalize(v)
    assert math.isclose(nx, ex, abs_tol=1e-6)
    assert math.isclose(ny, ey, abs_tol=1e-6)

def test_normalize_result_is_unit():
    for v in [(1, 1), (3, 4), (-5, 12), (0.1, 0.2)]:
        nx, ny = angles.normalize(v)
        assert math.isclose(math.hypot(nx, ny), 1.0, abs_tol=1e-6)


# --- wrap_signed_angle (extended) ---

@pytest.mark.parametrize("angle,expected", [
    (0.0, 0.0),
    (181.0, -179.0),
    (-181.0, 179.0),
    (360.0, 0.0),
    (-360.0, 0.0),
    (540.0, 180.0),
    (-540.0, -180.0),
])
def test_wrap_signed_angle_extended(angle, expected):
    assert math.isclose(angles.wrap_signed_angle(angle), expected, abs_tol=1e-6)


# --- pelvic_incidence_deg ---

def test_pelvic_incidence_deg_perpendicular():
    # vectors at 90° → angle=90 → PI=|90-90|=0
    assert math.isclose(angles.pelvic_incidence_deg((1, 0), (0, 1)), 0.0, abs_tol=1e-6)

def test_pelvic_incidence_deg_parallel():
    # parallel → angle=0 → PI=|90-0|=90
    assert math.isclose(angles.pelvic_incidence_deg((1, 0), (1, 0)), 90.0, abs_tol=1e-6)

def test_pelvic_incidence_deg_antiparallel():
    # antiparallel → angle=180 → PI=|90-180|=90
    assert math.isclose(angles.pelvic_incidence_deg((1, 0), (-1, 0)), 90.0, abs_tol=1e-6)

def test_pelvic_incidence_deg_nonnegative():
    for v1, v2 in [((1, 1), (2, -1)), ((0, 1), (1, 1)), ((-1, 0), (0, -1))]:
        assert angles.pelvic_incidence_deg(v1, v2) >= 0.0


# --- lumbosacral_lordosis_deg ---

def test_lumbosacral_lordosis_deg_nonzero():
    # v_L1=(1,0): slope=0°; v_S1=(1,1): slope=-45°; LL=-45-0=-45°
    assert math.isclose(angles.lumbosacral_lordosis_deg((1, 0), (1, 1)), -45.0, abs_tol=1e-6)

def test_lumbosacral_lordosis_deg_symmetric():
    # swapping v_L1 and v_S1 should negate LL
    ll_fwd = angles.lumbosacral_lordosis_deg((1, 0), (1, 1))
    ll_bwd = angles.lumbosacral_lordosis_deg((1, 1), (1, 0))
    assert math.isclose(ll_fwd, -ll_bwd, abs_tol=1e-6)

def test_lumbosacral_lordosis_deg_in_range():
    result = angles.lumbosacral_lordosis_deg((1, 0), (0, 1))
    assert -180.0 <= result <= 180.0


# --- l1_pelvic_angle_deg ---

def test_l1_pelvic_angle_deg_collinear_zero():
    # L1_center on the FH→S1_mid line → angle = 0
    FH = (0, 10)
    S1_mid = (0, 0)
    L1_center_on_axis = (0, 5)
    assert math.isclose(angles.l1_pelvic_angle_deg(FH, S1_mid, L1_center_on_axis), 0.0, abs_tol=1e-6)

def test_l1_pelvic_angle_deg_symmetric():
    # Placing L1_center symmetrically on each side should give equal magnitude, opposite sign
    FH = (0, 10)
    S1_mid = (0, 0)
    pos = angles.l1_pelvic_angle_deg(FH, S1_mid, (2, 5))
    neg = angles.l1_pelvic_angle_deg(FH, S1_mid, (-2, 5))
    assert math.isclose(pos, -neg, abs_tol=1e-6)

def test_l1_pelvic_angle_deg_in_range():
    FH, S1_mid = (0, 10), (0, 0)
    for x in [-5, -1, 0, 1, 5]:
        result = angles.l1_pelvic_angle_deg(FH, S1_mid, (x, 5))
        assert -180.0 <= result <= 180.0


# --- compute_angles_from_points (additional) ---

def test_compute_angles_error_message_lists_missing_keys():
    with pytest.raises(ValueError, match="L1_ant"):
        angles.compute_angles_from_points({"FH": (0, 0)})

def test_compute_angles_all_five_keys_present():
    pts = {
        "FH": (0, 10), "S1_ant": (-0.5, 0), "S1_post": (0.5, 0),
        "L1_ant": (-0.5, 5), "L1_post": (0.5, 5),
    }
    result = angles.compute_angles_from_points(pts)
    assert {"PI", "PT", "SS", "LL"} == set(result.keys())

def test_compute_angles_six_keys_includes_l1pa():
    pts = {
        "FH": (0, 10), "S1_ant": (-0.5, 0), "S1_post": (0.5, 0),
        "L1_ant": (-0.5, 5), "L1_post": (0.5, 5), "L1_center": (1.0, 5),
    }
    result = angles.compute_angles_from_points(pts)
    assert set(result.keys()) == {"PI", "PT", "SS", "LL", "L1PA"}

def test_compute_angles_pi_nonneg():
    pts = {
        "FH": (0.5, 2.0), "S1_ant": (0.0, 0.0), "S1_post": (1.0, 1.0),
        "L1_ant": (0.0, 1.0), "L1_post": (1.0, 2.0),
    }
    result = angles.compute_angles_from_points(pts)
    assert result["PI"] >= 0.0

def test_compute_angles_ss_matches_s1_slope():
    # Horizontal S1: v_S1=(1,0) → SS should be 0
    pts = {
        "FH": (0, 5), "S1_ant": (0.0, 0.0), "S1_post": (1.0, 0.0),
        "L1_ant": (0.0, 3.0), "L1_post": (1.0, 3.0),
    }
    result = angles.compute_angles_from_points(pts)
    assert math.isclose(result["SS"], 0.0, abs_tol=1e-6)
