"""train/landmark_scheme.py の単体テスト。"""

import pytest
from train.landmark_scheme import (
    ALL_LANDMARK_KEYS,
    BASE_LANDMARKS,
    BBOX_CLASSES,
    BBOX_CLASS_TO_IDX,
    CERVICAL,
    FULL_4PT_VERTEBRAE,
    LUMBAR,
    LUMBAR_VARIANTS,
    OPTIONAL_LANDMARKS,
    REGION_CHANNELS,
    REGION_VALID_CHANNELS,
    THORACIC,
    derive_bboxes,
    get_region_for_landmark,
    is_optional,
    make_landmark_template,
)


class TestLandmarkCounts:
    def test_base_landmark_count(self):
        # EAC(1) + C2×2 + 22椎体×4 + S1×2 + FH+femur×2 = 1+2+88+2+3 = 96
        assert len(BASE_LANDMARKS) == 96

    def test_optional_landmark_count(self):
        assert len(OPTIONAL_LANDMARKS) == 4  # L6_1〜L6_4

    def test_all_landmark_count(self):
        assert len(ALL_LANDMARK_KEYS) == 100

    def test_no_duplicates(self):
        assert len(ALL_LANDMARK_KEYS) == len(set(ALL_LANDMARK_KEYS))

    def test_vertebra_groups(self):
        assert len(CERVICAL) == 5    # C3-C7
        assert len(THORACIC) == 12   # T1-T12
        assert len(LUMBAR) == 5      # L1-L5
        assert len(FULL_4PT_VERTEBRAE) == 22

    def test_4pt_vertebrae_keys(self):
        for v in FULL_4PT_VERTEBRAE:
            for i in range(1, 5):
                assert f"{v}_{i}" in ALL_LANDMARK_KEYS


class TestSpecialLandmarks:
    def test_eac_present(self):
        assert "EAC" in BASE_LANDMARKS

    def test_c2_inferior_only(self):
        assert "C2_3" in BASE_LANDMARKS
        assert "C2_4" in BASE_LANDMARKS
        assert "C2_1" not in ALL_LANDMARK_KEYS
        assert "C2_2" not in ALL_LANDMARK_KEYS

    def test_s1_superior_only(self):
        assert "S1_1" in BASE_LANDMARKS
        assert "S1_2" in BASE_LANDMARKS
        assert "S1_3" not in ALL_LANDMARK_KEYS
        assert "S1_4" not in ALL_LANDMARK_KEYS

    def test_femur_landmarks(self):
        assert "FH" in BASE_LANDMARKS
        assert "femur_prox" in BASE_LANDMARKS
        assert "femur_dist" in BASE_LANDMARKS

    def test_l6_optional(self):
        for i in range(1, 5):
            key = f"L6_{i}"
            assert key in OPTIONAL_LANDMARKS
            assert is_optional(key)
            assert not is_optional("L5_1")


class TestBBoxClasses:
    def test_bbox_class_count(self):
        # skull, C2, C3-C7(5), T1-T12(12), L1-L5(5), L6, S1, pelvis = 27
        assert len(BBOX_CLASSES) == 27

    def test_bbox_class_to_idx_consistent(self):
        for cls, idx in BBOX_CLASS_TO_IDX.items():
            assert BBOX_CLASSES[idx] == cls

    def test_skull_and_pelvis_present(self):
        assert "skull" in BBOX_CLASSES
        assert "pelvis" in BBOX_CLASSES


class TestRegionChannels:
    def test_all_regions_have_4_channels(self):
        for region, channels in REGION_CHANNELS.items():
            assert len(channels) == 4, f"{region} should have 4 channels"

    def test_skull_ch0_is_eac(self):
        assert REGION_CHANNELS["skull"][0] == "EAC"
        assert all(c is None for c in REGION_CHANNELS["skull"][1:])

    def test_c2_uses_ch2_ch3(self):
        assert REGION_CHANNELS["C2"][2] == "C2_3"
        assert REGION_CHANNELS["C2"][3] == "C2_4"
        assert REGION_CHANNELS["C2"][0] is None
        assert REGION_CHANNELS["C2"][1] is None

    def test_s1_uses_ch0_ch1(self):
        assert REGION_CHANNELS["S1"][0] == "S1_1"
        assert REGION_CHANNELS["S1"][1] == "S1_2"
        assert REGION_CHANNELS["S1"][2] is None
        assert REGION_CHANNELS["S1"][3] is None

    def test_pelvis_channels(self):
        assert REGION_CHANNELS["pelvis"][0] == "FH"
        assert REGION_CHANNELS["pelvis"][1] == "femur_prox"
        assert REGION_CHANNELS["pelvis"][2] == "femur_dist"
        assert REGION_CHANNELS["pelvis"][3] is None

    def test_4pt_vertebra_uses_all_channels(self):
        for v in FULL_4PT_VERTEBRAE:
            for i in range(1, 5):
                assert REGION_CHANNELS[v][i - 1] == f"{v}_{i}"

    def test_valid_channels_subset_of_all_keys(self):
        all_keys_set = set(ALL_LANDMARK_KEYS)
        for region, channels in REGION_CHANNELS.items():
            for key in channels:
                if key is not None:
                    assert key in all_keys_set, f"{key} not in ALL_LANDMARK_KEYS"

    def test_region_valid_channels(self):
        assert REGION_VALID_CHANNELS["skull"] == [0]
        assert REGION_VALID_CHANNELS["C2"] == [2, 3]
        assert REGION_VALID_CHANNELS["S1"] == [0, 1]
        assert REGION_VALID_CHANNELS["pelvis"] == [0, 1, 2]
        for v in FULL_4PT_VERTEBRAE:
            assert REGION_VALID_CHANNELS[v] == [0, 1, 2, 3]


class TestMakeLandmarkTemplate:
    def test_has_all_keys(self):
        tmpl = make_landmark_template()
        assert set(tmpl.keys()) == set(ALL_LANDMARK_KEYS)

    def test_all_values_null(self):
        tmpl = make_landmark_template()
        for key, val in tmpl.items():
            assert val["i"] is None
            assert val["j"] is None
            assert val["k"] == 0


class TestDeriveBboxes:
    def _make_lm(self, entries: dict) -> dict:
        lm: dict = {k: None for k in ALL_LANDMARK_KEYS}
        lm.update(entries)
        return lm

    def test_returns_none_for_all_null(self):
        lm = {k: None for k in ALL_LANDMARK_KEYS}
        boxes = derive_bboxes(lm, (1000, 800))
        for region in REGION_CHANNELS:
            assert boxes[region] is None

    def test_4pt_vertebra_bbox(self):
        lm = self._make_lm({
            "L3_1": (200.0, 300.0),
            "L3_2": (150.0, 300.0),
            "L3_3": (150.0, 360.0),
            "L3_4": (200.0, 360.0),
        })
        boxes = derive_bboxes(lm, (1000, 800))
        box = boxes["L3"]
        assert box is not None
        x1, y1, x2, y2 = box
        # L3のi: 150-200, j: 300-360
        assert x1 < 150
        assert y1 < 300
        assert x2 > 200
        assert y2 > 360

    def test_bbox_within_image_bounds(self):
        # 端に近いランドマーク
        lm = self._make_lm({
            "T1_1": (5.0, 5.0),
            "T1_2": (3.0, 5.0),
            "T1_3": (3.0, 15.0),
            "T1_4": (5.0, 15.0),
        })
        boxes = derive_bboxes(lm, (100, 100))
        box = boxes["T1"]
        assert box is not None
        x1, y1, x2, y2 = box
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 99
        assert y2 <= 99

    def test_eac_skull_bbox(self):
        lm = self._make_lm({"EAC": (400.0, 50.0)})
        boxes = derive_bboxes(lm, (3000, 2500))
        box = boxes["skull"]
        assert box is not None
        x1, y1, x2, y2 = box
        assert x1 < 400
        assert x2 > 400
        assert y1 < 50
        assert y2 > 50

    def test_pelvis_bbox_from_3pts(self):
        lm = self._make_lm({
            "FH": (1200.0, 2500.0),
            "femur_prox": (1300.0, 2600.0),
            "femur_dist": (1350.0, 2800.0),
        })
        boxes = derive_bboxes(lm, (3000, 2500))
        box = boxes["pelvis"]
        assert box is not None
        x1, y1, x2, y2 = box
        assert x1 < 1200
        assert y1 < 2500
        assert x2 > 1350
        assert y2 > 2800

    def test_lumbar_variant_l6_optional(self):
        # L6 が存在しない場合は None
        lm = {k: None for k in ALL_LANDMARK_KEYS}
        boxes = derive_bboxes(lm, (1000, 800))
        assert boxes["L6"] is None

        # L6 が存在する場合はBBox生成
        lm["L6_1"] = (200.0, 700.0)
        lm["L6_2"] = (150.0, 700.0)
        lm["L6_3"] = (150.0, 760.0)
        lm["L6_4"] = (200.0, 760.0)
        boxes = derive_bboxes(lm, (1000, 800))
        assert boxes["L6"] is not None


class TestGetRegionForLandmark:
    def test_eac_maps_to_skull(self):
        assert get_region_for_landmark("EAC") == "skull"

    def test_fh_maps_to_pelvis(self):
        assert get_region_for_landmark("FH") == "pelvis"

    def test_c2_3_maps_to_c2(self):
        assert get_region_for_landmark("C2_3") == "C2"

    def test_vertebra_key_maps_to_vertebra(self):
        assert get_region_for_landmark("T5_2") == "T5"
        assert get_region_for_landmark("L3_4") == "L3"

    def test_unknown_key_returns_none(self):
        assert get_region_for_landmark("UNKNOWN") is None


class TestLumbarVariants:
    def test_variants_defined(self):
        assert "normal" in LUMBAR_VARIANTS
        assert "lumbarization" in LUMBAR_VARIANTS
        assert "sacralization" in LUMBAR_VARIANTS
