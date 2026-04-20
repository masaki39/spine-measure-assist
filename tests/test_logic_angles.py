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


def test_lumbosacral_lordosis_deg_same_slope():
    assert math.isclose(angles.lumbosacral_lordosis_deg((1, 0), (1, 0)), 0.0, abs_tol=1e-6)


def test_signed_slope_angle_deg_zero_raises():
    with pytest.raises(ValueError):
        angles.signed_slope_angle_deg((0, 0))


def test_signed_vertical_angle_deg_zero_raises():
    with pytest.raises(ValueError):
        angles.signed_vertical_angle_deg((0, 0))
