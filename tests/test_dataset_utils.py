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


# --- _percentile_clip_norm (additional) ---

def test_percentile_clip_norm_constant_image():
    img = np.full((20, 20), 5.0, dtype=np.float32)
    result = _percentile_clip_norm(img)
    assert result.dtype == np.float32
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_percentile_clip_norm_preserves_shape():
    img = np.random.default_rng(0).random((32, 48)).astype(np.float32)
    result = _percentile_clip_norm(img)
    assert result.shape == (32, 48)


# --- _make_heatmaps (additional) ---

def test_make_heatmaps_max_value_is_one():
    coords = [(16.0, 16.0)]
    hm = _make_heatmaps(coords, size=(32, 32), sigma=2.0)
    assert abs(hm[0].max().item() - 1.0) < 1e-3


def test_make_heatmaps_single_channel_peak_matches_coord():
    coords = [(10.0, 5.0)]  # x=10, y=5
    hm = _make_heatmaps(coords, size=(32, 32), sigma=1.5)
    idx = hm[0].argmax().item()
    peak_y, peak_x = divmod(idx, 32)
    assert peak_x == 10
    assert peak_y == 5


def test_make_heatmaps_non_square():
    coords = [(20.0, 10.0), (5.0, 30.0)]
    hm = _make_heatmaps(coords, size=(64, 32), sigma=2.0)
    assert hm.shape == (2, 64, 32)


# --- _resize_with_padding (additional) ---

def test_resize_with_padding_square_image_no_padding():
    img = torch.ones(1, 64, 64)
    _, scale, pad_x, pad_y = _resize_with_padding(img, (64, 64))
    assert abs(scale - 1.0) < 1e-3
    assert pad_x == 0
    assert pad_y == 0


def test_resize_with_padding_tall_image_pad_x_positive():
    img = torch.ones(1, 200, 100)
    _, scale, pad_x, _ = _resize_with_padding(img, (128, 128))
    new_w = int(round(100 * scale))
    assert pad_x == (128 - new_w) // 2


def test_resize_with_padding_wide_image_pad_y_positive():
    img = torch.ones(1, 100, 200)
    _, scale, _, pad_y = _resize_with_padding(img, (128, 128))
    new_h = int(round(100 * scale))
    assert pad_y == (128 - new_h) // 2


# --- HeatmapDataset full pipeline ---

def _write_full_sample(tmp_path):
    import json
    img = np.zeros((100, 80), dtype=np.float32)
    img[10, 20] = 1.0
    np.save(str(tmp_path / "K001_image.npy"), img)
    landmarks = {
        name: {"i": float(x), "j": float(y), "k": 0.0}
        for name, x, y in [
            ("L1_ant",    10.0, 20.0),
            ("L1_post",   20.0, 20.0),
            ("S1_ant",    10.0, 80.0),
            ("S1_post",   20.0, 80.0),
            ("FH",        40.0, 90.0),
            ("L1_center", 15.0, 20.0),
        ]
    }
    with open(tmp_path / "K001_landmarks.json", "w") as f:
        json.dump({"landmarks_ijk": landmarks, "metadata": {}}, f)


def test_dataset_full_pipeline_output_keys(tmp_path):
    _write_full_sample(tmp_path)
    ds = HeatmapDataset(data_dir=str(tmp_path), resize=(128, 128), sigma=3.0)
    sample = ds[0]
    assert "image" in sample
    assert "heatmap" in sample
    assert "coords" in sample
    assert "case_id" in sample


def test_dataset_full_pipeline_shapes(tmp_path):
    _write_full_sample(tmp_path)
    ds = HeatmapDataset(data_dir=str(tmp_path), resize=(128, 128), sigma=3.0)
    sample = ds[0]
    assert sample["image"].shape == (1, 128, 128)
    assert sample["heatmap"].shape == (6, 128, 128)
    assert sample["coords"].shape == (6, 2)


def test_dataset_full_pipeline_case_id(tmp_path):
    _write_full_sample(tmp_path)
    ds = HeatmapDataset(data_dir=str(tmp_path), resize=(128, 128), sigma=3.0)
    assert ds[0]["case_id"] == "K001"


def test_dataset_full_pipeline_image_range(tmp_path):
    _write_full_sample(tmp_path)
    ds = HeatmapDataset(data_dir=str(tmp_path), resize=(128, 128), sigma=3.0)
    img = ds[0]["image"]
    assert img.min() >= 0.0
    assert img.max() <= 1.0


def test_dataset_3d_image_uses_first_slice(tmp_path):
    import json
    img_3d = np.zeros((3, 64, 64), dtype=np.float32)
    img_3d[0, 10, 10] = 1.0
    np.save(str(tmp_path / "K002_image.npy"), img_3d)
    landmarks = {name: {"i": 10.0, "j": 10.0, "k": 0.0} for name in LANDMARK_ORDER}
    with open(tmp_path / "K002_landmarks.json", "w") as f:
        json.dump({"landmarks_ijk": landmarks, "metadata": {}}, f)
    ds = HeatmapDataset(data_dir=str(tmp_path), resize=(64, 64), sigma=2.0)
    sample = ds[0]
    assert sample["image"].shape == (1, 64, 64)


def test_dataset_discover_samples_empty_dir_raises(tmp_path):
    with pytest.raises(RuntimeError, match="No samples found"):
        HeatmapDataset(data_dir=str(tmp_path))


def test_dataset_len(tmp_path):
    _write_full_sample(tmp_path)
    ds = HeatmapDataset(data_dir=str(tmp_path), resize=(64, 64), sigma=2.0)
    assert len(ds) == 1
