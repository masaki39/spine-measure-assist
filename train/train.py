"""
Minimal training loop for sagittal landmark heatmaps.
Expect data_dir to contain *_image.npy and *_landmarks.json exported from Slicer.
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import HeatmapDataset, LANDMARK_ORDER
from model import SmallUNet, get_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Folder with *_image.npy and *_landmarks.json (exported from Slicer)")
    p.add_argument("--save-dir", default="train/runs", help="Where to save checkpoints (best.pt / last.pt)")
    p.add_argument("--epochs", type=int, default=20, help="How many passes over the dataset (more can improve accuracy but takes time)")
    p.add_argument("--batch-size", type=int, default=4, help="How many samples processed together in one step (fits GPU/CPU memory)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (step size for optimization)")
    p.add_argument("--resize", type=int, nargs=2, default=[512, 512], metavar=("H", "W"), help="Target size after aspect-ratio padding")
    p.add_argument("--sigma", type=float, default=15.0, help="Gaussian sigma (px) for landmark heatmaps; larger spreads targets wider")
    p.add_argument("--num-workers", type=int, default=2, help="Data loading threads (increase if CPU has cores to spare)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint to resume/finetune from")
    p.add_argument("--landmarks", default=None, help="Comma-separated landmark names (default: L1_ant,L1_post,S1_ant,S1_post,FH)")
    p.add_argument("--backbone", default="smallunet", help="Model backbone: 'smallunet' or segmentation_models_pytorch encoder (e.g. 'resnet34', 'efficientnet-b3')")
    p.add_argument("--augment", action="store_true", help="Enable data augmentation (rotation, elastic transform, brightness/contrast)")
    p.add_argument("--loss", default="mse", choices=["mse", "awl"], help="Loss function: mse (default) or awl (Adaptive Wing Loss)")
    p.add_argument("--split-seed", type=int, default=42, help="Random seed for reproducible train/val/test split")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of data for validation (default: 0.1)")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of data held out for final test (default: 0.1)")
    return p.parse_args()


def adaptive_wing_loss(pred: torch.Tensor, target: torch.Tensor,
                        omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1) -> torch.Tensor:
    """Adaptive Wing Loss for heatmap-based landmark detection (Wang et al., ICCV 2019)."""
    delta = (target - pred).abs()
    # clamp alpha-target away from 0 for numerical safety
    amt = (alpha - target).clamp(min=0.1)
    A = omega * amt / epsilon * torch.pow(theta / epsilon, amt - 1) / (
        1 + torch.pow(theta / epsilon, amt)
    )
    C = theta * A - omega * torch.log(1 + torch.pow(theta / epsilon, amt))
    # clamp delta to avoid 0^p gradient issues in log branch
    loss = torch.where(
        delta < theta,
        omega * torch.log(1 + torch.pow(delta.clamp(min=1e-12) / epsilon, amt)),
        A * delta - C,
    )
    return loss.mean()


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        img = batch["image"].to(device)
        target = batch["heatmap"].to(device)
        pred = model(img)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * img.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            img = batch["image"].to(device)
            target = batch["heatmap"].to(device)
            pred = model(img)
            loss = loss_fn(pred, target)
            total_loss += loss.item() * img.size(0)
    return total_loss / len(loader.dataset)


def _build_model_with_landmark_transfer(ckpt, old_landmarks, new_landmarks, old_backbone, new_backbone, device):
    """Transfer backbone weights + matched head channels when landmark set changes (SmallUNet only)."""
    new_model = get_model(new_backbone, num_landmarks=len(new_landmarks)).to(device)
    if old_backbone != "smallunet" or new_backbone != "smallunet":
        return new_model

    old_state = ckpt["model_state"]
    new_state = new_model.state_dict()

    for key in new_state:
        if not key.startswith("head.") and key in old_state:
            new_state[key] = old_state[key]

    for new_idx, name in enumerate(new_landmarks):
        if name in old_landmarks:
            old_idx = old_landmarks.index(name)
            new_state["head.weight"][new_idx] = old_state["head.weight"][old_idx]
            new_state["head.bias"][new_idx] = old_state["head.bias"][old_idx]

    new_model.load_state_dict(new_state)
    return new_model


def main():
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    landmark_keys = [k.strip() for k in args.landmarks.split(",")] if args.landmarks else LANDMARK_ORDER

    # Create train/val datasets separately so augmentation applies only to training
    train_dataset = HeatmapDataset(
        data_dir=args.data_dir,
        resize=tuple(args.resize),
        sigma=args.sigma,
        landmark_keys=landmark_keys,
        augment=args.augment,
    )
    val_dataset = HeatmapDataset(
        data_dir=args.data_dir,
        resize=tuple(args.resize),
        sigma=args.sigma,
        landmark_keys=landmark_keys,
        augment=False,
    )

    n_total = len(train_dataset)
    if n_total == 1:
        print("WARNING: サンプルが1件のみ。訓練データを検証・テストにも使用します。")
        train_set = train_dataset
        val_set = val_dataset
        test_set = val_dataset
        splits_map = {"train": [0], "val": [0], "test": [0]}
    else:
        import random
        rng = random.Random(args.split_seed)
        all_indices = list(range(n_total))
        rng.shuffle(all_indices)

        n_test = max(1, int(n_total * args.test_ratio))
        n_val = max(1, int(n_total * args.val_ratio))
        n_train = n_total - n_val - n_test

        test_idx = all_indices[:n_test]
        val_idx = all_indices[n_test:n_test + n_val]
        train_idx = all_indices[n_test + n_val:]

        train_set = Subset(train_dataset, train_idx)
        val_set = Subset(val_dataset, val_idx)
        test_set = Subset(val_dataset, test_idx)

        # Map indices back to case IDs for splits.json
        all_samples = train_dataset.dataset.samples if hasattr(train_dataset, "dataset") else train_dataset.samples
        def idx_to_caseid(idxs):
            return [all_samples[i][0] for i in idxs]
        splits_map = {
            "train": idx_to_caseid(train_idx),
            "val": idx_to_caseid(val_idx),
            "test": idx_to_caseid(test_idx),
        }

        print(f"Split (seed={args.split_seed}): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    loss_fn = adaptive_wing_loss if args.loss == "awl" else lambda p, t: torch.mean((p - t) ** 2)

    model = get_model(args.backbone, num_landmarks=len(landmark_keys)).to(device)

    # Pretrained encoder uses 1/10 LR to preserve ImageNet features; random decoder uses full LR
    if args.backbone != "smallunet" and hasattr(model, "encoder"):
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.lr * 0.1},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": args.lr},
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        old_landmarks = ckpt.get("config", {}).get("landmarks") or LANDMARK_ORDER
        if isinstance(old_landmarks, str):
            old_landmarks = [k.strip() for k in old_landmarks.split(",")]
        old_backbone = ckpt.get("config", {}).get("backbone", "smallunet")

        if old_landmarks == landmark_keys and old_backbone == args.backbone:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print(f"ランドマーク構成またはバックボーンが変わりました。")
            print(f"  backbone: {old_backbone} → {args.backbone}")
            print(f"  landmarks: {old_landmarks} → {landmark_keys}")
            model = _build_model_with_landmark_transfer(
                ckpt, old_landmarks, landmark_keys, old_backbone, args.backbone, device
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, val {ckpt.get('val_loss', '?'):.4f})")

    # Save split info for reproducible test evaluation
    import json as _json
    splits_path = save_dir / "splits.json"
    with open(splits_path, "w") as fp:
        _json.dump(splits_map, fp, indent=2)
    print(f"Splits saved to {splits_path}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = validate(model, val_loader, device, loss_fn)
        print(f"[{epoch}/{args.epochs}] train {train_loss:.4f} | val {val_loss:.4f}")
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": {**vars(args), "landmarks": landmark_keys},
        }
        torch.save(ckpt, save_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  -> saved best.pt (val {val_loss:.4f})")

    # Final test evaluation on held-out set
    test_loss = validate(model, test_loader, device, loss_fn)
    print(f"\n=== Final test loss (held-out, N={len(test_set)}): {test_loss:.4f} ===")
    print(f"To evaluate angles/MRE on test set:")
    print(f"  uv run python train/infer_onnx.py --model {save_dir}/best.onnx --dir {args.data_dir} --splits {splits_path} --subset test")


if __name__ == "__main__":
    main()
