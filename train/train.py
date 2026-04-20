"""
Minimal training loop for sagittal landmark heatmaps.
Expect data_dir to contain *_image.npy and *_landmarks.json exported from Slicer.
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HeatmapDataset, LANDMARK_ORDER
from model import SmallUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Folder with *_image.npy and *_landmarks.json (exported from Slicer)")
    p.add_argument("--save-dir", default="runs", help="Where to save checkpoints (best.pt / last.pt)")
    p.add_argument("--epochs", type=int, default=20, help="How many passes over the dataset (more can improve accuracy but takes time)")
    p.add_argument("--batch-size", type=int, default=4, help="How many samples processed together in one step (fits GPU/CPU memory)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (step size for optimization)")
    p.add_argument("--resize", type=int, nargs=2, default=[512, 512], metavar=("H", "W"), help="Target size after aspect-ratio padding")
    p.add_argument("--sigma", type=float, default=15.0, help="Gaussian sigma (px) for landmark heatmaps; larger spreads targets wider")
    p.add_argument("--num-workers", type=int, default=2, help="Data loading threads (increase if CPU has cores to spare)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint to resume/finetune from")
    p.add_argument("--landmarks", default=None, help="Comma-separated landmark names (default: L1_ant,L1_post,S1_ant,S1_post,FH)")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        img = batch["image"].to(device)
        target = batch["heatmap"].to(device)
        pred = model(img)
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            img = batch["image"].to(device)
            target = batch["heatmap"].to(device)
            pred = model(img)
            loss = torch.mean((pred - target) ** 2)
            total_loss += loss.item() * img.size(0)
    return total_loss / len(loader.dataset)


def _build_model_with_landmark_transfer(ckpt, old_landmarks, new_landmarks, device):
    """
    Build SmallUNet with new_landmarks, transferring backbone + matched head channels from checkpoint.
    New landmark channels are randomly initialized.
    """
    new_model = SmallUNet(num_landmarks=len(new_landmarks)).to(device)
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

    dataset = HeatmapDataset(
        data_dir=args.data_dir,
        resize=tuple(args.resize),
        sigma=args.sigma,
        landmark_keys=landmark_keys,
    )
    # 90/10 split; 1サンプルのみの場合は訓練・検証共用
    n_total = len(dataset)
    if n_total == 1:
        print("WARNING: サンプルが1件のみ。訓練データを検証にも使用します。")
        train_set = dataset
        val_set = dataset
    else:
        n_val = max(1, n_total // 10)
        n_train = n_total - n_val
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SmallUNet(num_landmarks=len(landmark_keys)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        old_landmarks = ckpt.get("config", {}).get("landmarks") or LANDMARK_ORDER
        if isinstance(old_landmarks, str):
            old_landmarks = [k.strip() for k in old_landmarks.split(",")]

        if old_landmarks == landmark_keys:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print(f"ランドマーク構成が変わりました: {old_landmarks} → {landmark_keys}")
            print("エンコーダ/デコーダを移植し、一致するヘッドチャネルをコピーします。")
            model = _build_model_with_landmark_transfer(ckpt, old_landmarks, landmark_keys, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, val {ckpt.get('val_loss', '?'):.4f})")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
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


if __name__ == "__main__":
    main()
