import math

import numpy as np
import pytest

from logic_inference import OnnxInferenceLogic, _pad_resize, _percentile_clip_norm, _resize_bilinear


def test_percentile_clip_norm_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 1000, (100, 100)).astype(np.float32)
    result = _percentile_clip_norm(img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_percentile_clip_norm_dtype():
    img = np.ones((10, 10), dtype=np.float64)
    result = _percentile_clip_norm(img)
    assert result.dtype == np.float32


def test_resize_bilinear_shape():
    img = np.ones((100, 80), dtype=np.float32)
    result = _resize_bilinear(img, 50, 40)
    assert result.shape == (50, 40)


def test_resize_bilinear_constant_image():
    img = np.full((60, 40), 0.5, dtype=np.float32)
    result = _resize_bilinear(img, 30, 20)
    assert np.allclose(result, 0.5, atol=1e-5)


def test_pad_resize_output_shape():
    img = np.ones((200, 100), dtype=np.float32)
    result, _, _, _ = _pad_resize(img, (128, 128))
    assert result.shape == (128, 128)


def test_pad_resize_tall_image_scale():
    # H=200 > W=100, target=(128,128): scale = min(128/200, 128/100) = 128/200
    img = np.ones((200, 100), dtype=np.float32)
    _, scale, _, pad_y = _pad_resize(img, (128, 128))
    assert math.isclose(scale, 128 / 200, rel_tol=1e-3)
    assert pad_y == 0


def test_pad_resize_wide_image_scale():
    # H=100, W=200, target=(128,128): scale = min(128/100, 128/200) = 128/200
    img = np.ones((100, 200), dtype=np.float32)
    _, scale, pad_x, _ = _pad_resize(img, (128, 128))
    assert math.isclose(scale, 128 / 200, rel_tol=1e-3)
    assert pad_x == 0


def test_postprocess_coords():
    logic = OnnxInferenceLogic()
    logic.target_hw = (64, 64)
    # scale=1.0, pad_x=0, pad_y=0 -> coords == heatmap peak
    hm = np.zeros((1, 2, 64, 64), dtype=np.float32)
    hm[0, 0, 10, 20] = 1.0  # landmark0: y=10, x=20
    hm[0, 1, 30, 40] = 1.0  # landmark1: y=30, x=40
    coords = logic._postprocess(hm, scale=1.0, pad_x=0, pad_y=0)
    assert math.isclose(coords[0][0], 20.0, abs_tol=0.5)
    assert math.isclose(coords[0][1], 10.0, abs_tol=0.5)
    assert math.isclose(coords[1][0], 40.0, abs_tol=0.5)
    assert math.isclose(coords[1][1], 30.0, abs_tol=0.5)


def test_postprocess_with_padding():
    logic = OnnxInferenceLogic()
    # pad_x=10, pad_y=5, scale=2.0 -> x_orig=(x-10)/2, y_orig=(y-5)/2
    hm = np.zeros((1, 1, 64, 64), dtype=np.float32)
    hm[0, 0, 15, 30] = 1.0  # y=15, x=30
    coords = logic._postprocess(hm, scale=2.0, pad_x=10, pad_y=5)
    assert math.isclose(coords[0][0], (30 - 10) / 2.0, abs_tol=0.5)
    assert math.isclose(coords[0][1], (15 - 5) / 2.0, abs_tol=0.5)
