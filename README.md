# spine-measure-assist

3D Slicer toolkit for spine sagittal alignment measurement and AI model training on lateral spine X-rays (DICOM).

---

## Overview

This repository contains three components:

1. **Slicer Modules** (`slicer/`) — Interactive measurement tools as 3D Slicer scripted modules
2. **Training Pipeline** (`train/`) — Scripts for training and evaluating landmark detection models
3. **Utility Scripts** (`scripts/`) — Data extraction, evaluation, and analysis scripts

> Note: This project is under active development. APIs and file formats may change.

---

## Slicer Modules

Each module is a [3D Slicer](https://www.slicer.org/) scripted module that loads a lateral spine X-ray and assists with manual landmark placement and angle computation.

### LumbarMeasureAssist

Measures pelvic and lumbosacral parameters from 5 landmarks on lateral spine X-rays.

- **Landmarks (5):** L1_ant, L1_post, S1_ant, S1_post, FH
- **Outputs:** PI, PT, SS, LL (Lumbosacral Lordosis), L1PA

### CervicalMeasureAssist

Measures cervical alignment parameters from 8 landmarks on lateral cervical X-rays.

- **Landmarks (8):** C2_ant, C2_post, C2_center, C7_inf_ant, C7_inf_post, C7_sup_post, T1_ant, T1_post
- **Outputs:** C2C7 angle (°), T1 slope (°), C2C7 SVA (mm)

### WholeSpineAssist

Annotation tool for placing 96-point landmarks across the full spine (C2 to femur) for Phase 2 dataset construction.

- **Landmarks (~96):** All vertebral corners from C3–L5 plus C2, S1, skull, femur
- **Workflow:** Dataset-driven; keyboard-accelerated sequential landmark placement

### Setup (Slicer)

In 3D Slicer: `Developer Tools > Extension Wizard > Select Extension`

Select the `slicer/` directory of this repository. All three modules will be loaded at once.

---

## Training Pipeline

Training and evaluation scripts for landmark detection models. Uses a SmallUNet (Phase 1) and HRNet (Phase 2) architecture with heatmap regression.

### Workflow

```
train/train.py          Train SmallUNet (Phase 1)
train/export_lumbar.py  Export to ONNX
train/eval_lumbar.py    Evaluate: MRE, angle MAE, Bland-Altman, ICC

train/eval_cervical.py  Evaluate cervical model
train/eval_phase2.py    Evaluate two-stage Phase 2 model
```

### Commands

```bash
# Install ML dependencies
uv sync --extra ml

# Train (Phase 1 lumbar)
uv run python train/train.py \
  --data-dir /path/to/dataset/l1pa \
  --backbone smallunet --sigma 5 --augment --loss awl \
  --split-seed 42 --epochs 50

# Export to ONNX
uv run python train/export_lumbar.py \
  --checkpoint train/runs/best.pt --output train/runs/best.onnx

# Evaluate
uv run python train/eval_lumbar.py \
  --model train/runs/best.onnx --dir /path/to/dataset/l1pa

# Evaluate on test set only
uv run python train/eval_lumbar.py \
  --model train/runs/best.onnx --dir /path/to/dataset/l1pa \
  --splits train/runs/splits.json --subset test
```

GPU training notebooks are in `train/colab/` (Google Colab).

---

## Utility Scripts

```
scripts/
  extract_dataset.py          Extract WHOLE_SPINE LAT images from DB → Phase 2 dataset
  extract_cervical_dataset.py Extract cervical images from DB → cervical dataset
  inter_annotator_error.py    Compute MRE/MAE between two annotators (human baseline)
  merge_cervical_csv.py       Merge lateral/flexion/extension cervical CSVs into summary
  inspect_dicom.py            Inspect DICOM metadata
  export_angles_csv.py        Export computed angles to CSV
  extract_dicom_dir.py        Extract DICOMs from a directory
```

All path arguments accept `--help` for options. Hardcoded defaults point to the original data SSD.

```bash
uv run python scripts/extract_dataset.py --help
uv run python scripts/inter_annotator_error.py --root /path/to/dataset
```

---

## Development Setup

```bash
# Install dev dependencies
uv sync

# Install with ML dependencies (PyTorch, ONNX, etc.)
uv sync --extra ml

# Run tests (279 tests, no GPU required)
uv run -m pytest
```

---

## Project Structure

```
slicer/
  LumbarMeasureAssist/        Slicer module: lumbar/pelvic measurement
  CervicalMeasureAssist/      Slicer module: cervical measurement
  WholeSpineAssist/           Slicer module: full-spine annotation (Phase 2)
  shared/                     Shared UI components

train/
  train.py                    Training entry point (Phase 1)
  model.py                    SmallUNet architecture
  dataset.py                  HeatmapDataset (Phase 1)
  dataset_cervical.py         HeatmapDataset, cervical variant
  dataset_phase2.py           Dataset for Phase 2 (96-point)
  model_hrnet.py              HRNet-W32 + heatmap head (Phase 2)
  landmark_scheme.py          96-point landmark definition (single source of truth)
  eval_lumbar.py              Evaluate lumbar ONNX model
  eval_cervical.py            Evaluate cervical ONNX model
  eval_phase2.py              Evaluate two-stage Phase 2 model
  export_lumbar.py            Export lumbar model to ONNX
  export_phase2.py            Export Phase 2 model to ONNX
  colab/                      GPU training notebooks (Google Colab)
  runs/                       Model checkpoints and splits.json (gitignored)

scripts/                      Data preparation and analysis scripts

tests/                        Test suite (279 tests)
```
