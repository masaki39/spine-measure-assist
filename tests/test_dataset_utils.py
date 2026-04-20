import numpy as np
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train")))

from dataset import HeatmapDataset, LANDMARK_ORDER, _make_heatmaps, _percentile_clip_norm, _resize_with_padding


def test_percentile_clip_norm_range():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 1000, (80, 80)).astype(np.float32)
    result = _percentile_clip_norm(img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_percentile_clip_norm_dtype():
    img = np.ones((10, 10), dtype=np.float64)
    result = _percentile_clip_norm(img)
    assert result.dtype == np.float32


def test_make_heatmaps_shape():
    coords = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
    hm = _make_heatmaps(coords, size=(128, 128), sigma=3.0)
    assert hm.shape == (3, 128, 128)


def test_make_heatmaps_peak_location():
    coords = [(20.0, 10.0)]  # x=20, y=10
    hm = _make_heatmaps(coords, size=(64, 64), sigma=2.0)
    idx = hm[0].argmax().item()
    peak_y, peak_x = divmod(idx, 64)
    assert abs(peak_x - 20) <= 1
    assert abs(peak_y - 10) <= 1


def test_resize_with_padding_output_shape():
    img = torch.ones(1, 200, 100)
    result, _, _, _ = _resize_with_padding(img, (128, 128))
    assert result.shape == (1, 128, 128)


def test_resize_with_padding_tall_image():
    img = torch.ones(1, 200, 100)
    _, scale, _, pad_y = _resize_with_padding(img, (128, 128))
    assert abs(scale - 128 / 200) < 1e-3
    assert pad_y == 0


def test_resize_with_padding_wide_image():
    img = torch.ones(1, 100, 200)
    _, scale, pad_x, _ = _resize_with_padding(img, (128, 128))
    assert abs(scale - 128 / 200) < 1e-3
    assert pad_x == 0


def test_extract_coords_missing_key(tmp_path):
    import json
    npy = tmp_path / "case_image.npy"
    np.save(str(npy), np.ones((64, 64), dtype=np.float32))
    json_path = tmp_path / "case_landmarks.json"
    json_path.write_text(json.dumps({"landmarks_ijk": {"L1_ant": {"i": 10, "j": 10}}}))
    ds = HeatmapDataset.__new__(HeatmapDataset)
    ds.data_dir = str(tmp_path)
    ds.resize = (64, 64)
    ds.sigma = 3.0
    ds.percentile_clip = (1.0, 99.0)
    ds.landmark_keys = LANDMARK_ORDER
    ds.samples = [("case", str(npy), str(json_path))]
    with pytest.raises(ValueError, match="Missing landmark"):
        ds[0]


def test_extract_coords_missing_landmarks_ijk(tmp_path):
    import json
    npy = tmp_path / "case_image.npy"
    np.save(str(npy), np.ones((64, 64), dtype=np.float32))
    json_path = tmp_path / "case_landmarks.json"
    json_path.write_text(json.dumps({}))
    ds = HeatmapDataset.__new__(HeatmapDataset)
    ds.data_dir = str(tmp_path)
    ds.resize = (64, 64)
    ds.sigma = 3.0
    ds.percentile_clip = (1.0, 99.0)
    ds.landmark_keys = LANDMARK_ORDER
    ds.samples = [("case", str(npy), str(json_path))]
    with pytest.raises(ValueError, match="Missing landmarks_ijk"):
        ds[0]
