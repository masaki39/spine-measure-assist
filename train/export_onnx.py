"""
Export a trained checkpoint to ONNX for Slicer inference (CPU-friendly).
"""

import argparse
from pathlib import Path

import torch

from dataset import LANDMARK_ORDER
from model import SmallUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best.pt/last.pt")
    p.add_argument("--output", required=True, help="Path to output onnx file")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    resize = ckpt.get("config", {}).get("resize", [512, 512])
    height = args.height if args.height is not None else resize[0]
    width = args.width if args.width is not None else resize[1]

    model = SmallUNet(num_landmarks=len(LANDMARK_ORDER))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.zeros(1, 1, height, width, device=device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["heatmaps"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "heatmaps": {0: "batch"}},
    )
    print(f"Exported ONNX to {out_path} (input size: {height}x{width})")


if __name__ == "__main__":
    main()
