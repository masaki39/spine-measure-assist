"""
Stage 2 モデル（HRNet or SmallUNet）を ONNX にエクスポートする。

使用法:
  uv run python train/export_onnx_phase2.py \\
    --checkpoint train/runs/phase2_best.pt \\
    --output     train/runs/phase2_best.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.dataset_phase2 import CROP_SIZE
from train.model_hrnet import get_stage2_model


def export(checkpoint: Path, output: Path, backbone: str, opset: int = 17):
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    # checkpoint が {"model": state_dict, ...} 形式の場合に対応
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model = get_stage2_model(backbone=backbone, pretrained=False)
    model.load_state_dict(state)
    model.eval()

    H, W = CROP_SIZE
    dummy = torch.zeros(1, 1, H, W)
    with torch.no_grad():
        out = model(dummy)
    print(f"Model output shape: {out.shape}")

    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=opset,
        input_names=["image"],
        output_names=["heatmaps"],
        dynamic_axes={"image": {0: "batch"}, "heatmaps": {0: "batch"}},
    )
    print(f"Exported: {output}")


def main():
    parser = argparse.ArgumentParser(description="Export Phase2 Stage2 model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="出力 ONNX ファイルパス")
    parser.add_argument("--backbone", default="hrnet_w32", help="モデルバックボーン (hrnet_w32 / smallunet)")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.output)
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}")
    out.parent.mkdir(parents=True, exist_ok=True)

    export(ckpt, out, backbone=args.backbone, opset=args.opset)


if __name__ == "__main__":
    main()
