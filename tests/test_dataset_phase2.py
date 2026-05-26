"""train/dataset_phase2.py の単体テスト（ダミーデータ使用）。"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from train.dataset_phase2 import CROP_SIZE, Phase2Dataset
from train.landmark_scheme import (
    ALL_LANDMARK_KEYS,
    BBOX_CLASSES,
    FULL_4PT_VERTEBRAE,
    REGION_CHANNELS,
    make_landmark_template,
)


def _make_dummy_sample(
    out_dir: Path,
    case_id: str = "K001",
    annotated: bool = True,
    with_l6: bool = False,
) -> None:
    """ダミーの npy + json を生成する。"""
    H, W = 400, 300
    img = np.random.rand(H, W).astype(np.float32)
    np.save(str(out_dir / f"{case_id}_image.npy"), img)

    lm = make_landmark_template()
    if annotated:
        # 全椎体に合理的な座標を設定
        # 椎体列: i=100-200 (列), j = 20 + idx*12 (行)
        idx = 0
        for v in FULL_4PT_VERTEBRAE:
            j_top = 20 + idx * 12
            j_bot = j_top + 10
            lm[f"{v}_1"] = {"i": 160, "j": j_top, "k": 0}
            lm[f"{v}_2"] = {"i": 120, "j": j_top, "k": 0}
            lm[f"{v}_3"] = {"i": 120, "j": j_bot, "k": 0}
            lm[f"{v}_4"] = {"i": 160, "j": j_bot, "k": 0}
            idx += 1

        lm["C2_3"] = {"i": 120, "j": 10, "k": 0}
        lm["C2_4"] = {"i": 160, "j": 10, "k": 0}
        lm["S1_1"] = {"i": 160, "j": 310, "k": 0}
        lm["S1_2"] = {"i": 120, "j": 310, "k": 0}
        lm["EAC"]  = {"i": 140, "j": 5,  "k": 0}
        lm["FH"]         = {"i": 140, "j": 350, "k": 0}
        lm["femur_prox"] = {"i": 145, "j": 360, "k": 0}
        lm["femur_dist"] = {"i": 150, "j": 380, "k": 0}

        if with_l6:
            lm["L6_1"] = {"i": 160, "j": 290, "k": 0}
            lm["L6_2"] = {"i": 120, "j": 290, "k": 0}
            lm["L6_3"] = {"i": 120, "j": 300, "k": 0}
            lm["L6_4"] = {"i": 160, "j": 300, "k": 0}

    meta = {
        "case_id": case_id,
        "db_id": 1,
        "image_shape": [H, W],
        "metadata": {"spacing": [0.429, 0.429, 1.0], "ijk_to_ras": [[-0.429,0,0],[0,-0.429,0],[0,0,1]], "origin_ras": [0,0,0]},
        "lumbar_variant": "normal" if not with_l6 else "lumbarization",
        "landmarks_ijk": lm,
    }
    with open(str(out_dir / f"{case_id}_landmarks.json"), "w") as f:
        json.dump(meta, f)


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    _make_dummy_sample(tmp_path, "K001", annotated=True)
    _make_dummy_sample(tmp_path, "K002", annotated=True)
    _make_dummy_sample(tmp_path, "K003", annotated=False)  # 未アノテーション
    return str(tmp_path)


@pytest.fixture
def tmp_dataset_dir_l6(tmp_path):
    _make_dummy_sample(tmp_path, "K001", annotated=True, with_l6=True)
    return str(tmp_path)


class TestPhase2DatasetFull:
    def test_len(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="full", resize=(128, 128))
        assert len(ds) == 3

    def test_image_shape(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="full", resize=(128, 128))
        item = ds[0]
        assert item["image"].shape == (1, 128, 128)

    def test_heatmap_shape(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="full", resize=(128, 128))
        item = ds[0]
        assert item["heatmap"].shape == (100, 128, 128)

    def test_valid_mask_shape(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="full", resize=(128, 128))
        item = ds[0]
        assert item["valid_mask"].shape == (100,)
        assert item["valid_mask"].dtype == torch.bool

    def test_annotated_sample_has_some_valid(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="full", resize=(128, 128))
        # K001: annotated
        item = ds[0]
        assert item["valid_mask"].any()

    def test_unannotated_sample_all_invalid(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="full", resize=(128, 128))
        # K003: unannotated (sorted: K001=0, K002=1, K003=2)
        item = ds[2]
        assert not item["valid_mask"].any()

    def test_case_id_field(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="full", resize=(128, 128))
        item = ds[0]
        assert isinstance(item["case_id"], str)


class TestPhase2DatasetStage1:
    def test_labels_shape(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="stage1", resize=(128, 128))
        item = ds[0]
        labels = item["labels"]
        assert labels.ndim == 2
        assert labels.shape[1] == 5  # [cls, cx, cy, w, h]

    def test_labels_class_ids_valid(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="stage1", resize=(128, 128))
        item = ds[0]
        if len(item["labels"]) > 0:
            cls_ids = item["labels"][:, 0].long()
            assert (cls_ids >= 0).all()
            assert (cls_ids < len(BBOX_CLASSES)).all()

    def test_labels_normalized(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="stage1", resize=(128, 128))
        item = ds[0]
        if len(item["labels"]) > 0:
            coords = item["labels"][:, 1:]
            assert (coords >= 0).all()
            assert (coords <= 1).all()

    def test_unannotated_has_no_labels(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="stage1", resize=(128, 128))
        item = ds[2]  # K003: unannotated
        assert len(item["labels"]) == 0


class TestPhase2DatasetStage2:
    def test_len_is_samples_times_regions(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="stage2", resize=(128, 128))
        n_samples = 3
        n_regions = len(REGION_CHANNELS)
        assert len(ds) == n_samples * n_regions

    def test_crop_image_shape(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="stage2", resize=(128, 128))
        item = ds[0]
        assert item["image"].shape == (1, *CROP_SIZE)

    def test_heatmap_4channels(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="stage2", resize=(128, 128))
        item = ds[0]
        assert item["heatmap"].shape == (4, *CROP_SIZE)

    def test_valid_mask_4channels(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="stage2", resize=(128, 128))
        item = ds[0]
        assert item["valid_mask"].shape == (4,)
        assert item["valid_mask"].dtype == torch.bool

    def test_region_field_present(self, tmp_dataset_dir):
        ds = Phase2Dataset(tmp_dataset_dir, mode="stage2", resize=(128, 128))
        item = ds[0]
        assert "region" in item
        assert item["region"] in REGION_CHANNELS

    def test_region_filter(self, tmp_dataset_dir):
        ds = Phase2Dataset(
            tmp_dataset_dir,
            mode="stage2",
            resize=(128, 128),
            region_filter=["L3", "L4"],
        )
        assert len(ds) == 3 * 2  # 3 samples × 2 regions
        regions = {ds[i]["region"] for i in range(len(ds))}
        assert regions == {"L3", "L4"}

    def test_unannotated_bbox_invalid(self, tmp_dataset_dir):
        ds = Phase2Dataset(
            tmp_dataset_dir,
            mode="stage2",
            resize=(128, 128),
            region_filter=["L3"],
        )
        # K003 (idx=2) は未アノテーション
        item = ds[2]
        assert not item["bbox_valid"]
        assert not item["valid_mask"].any()


class TestL6OptionalHandling:
    def test_l6_bbox_generated_when_annotated(self, tmp_dataset_dir_l6):
        ds = Phase2Dataset(
            tmp_dataset_dir_l6,
            mode="stage1",
            resize=(128, 128),
        )
        item = ds[0]
        labels = item["labels"]
        # L6 が BBOX_CLASSES に含まれている
        from train.landmark_scheme import BBOX_CLASS_TO_IDX
        l6_cls = BBOX_CLASS_TO_IDX.get("L6")
        if l6_cls is not None and len(labels) > 0:
            cls_ids = labels[:, 0].long().tolist()
            assert l6_cls in cls_ids

    def test_stage2_l6_region(self, tmp_dataset_dir_l6):
        ds = Phase2Dataset(
            tmp_dataset_dir_l6,
            mode="stage2",
            resize=(128, 128),
            region_filter=["L6"],
        )
        item = ds[0]
        assert item["region"] == "L6"
        assert item["bbox_valid"]
        assert item["valid_mask"].all()
