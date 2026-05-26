"""
Stage 2 用 HRNet-W32 + heatmap head モデル。

timm 経由で HRNet-W32 を使用する（segmentation-models-pytorch が timm に依存）。
入力: (B, 1, 256, 256) グレースケール crop
出力: (B, 4, 256, 256) heatmap（チャネル意味は REGION_CHANNELS 参照）
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HRNetLandmark(nn.Module):
    """
    HRNet-W32 backbone + heatmap head。
    1ch グレースケール入力を 3ch に repeat して HRNet に渡す。
    """

    def __init__(self, num_channels: int = 4, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm が必要です: uv sync --extra ml")

        self.backbone = timm.create_model(
            "hrnet_w32",
            pretrained=pretrained,
            features_only=False,
            num_classes=0,   # classification head を除去
        )

        # HRNet-W32 の出力チャネル数を確認して head を構築
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            feat = self.backbone.forward_features(dummy)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            feat_channels = feat.shape[1]

        self.head = nn.Sequential(
            nn.Conv2d(feat_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_channels, kernel_size=1),
        )

        # 出力サイズが入力の 1/4 の場合は upsample
        self._need_upsample = feat.shape[-1] < 256 // 4 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,1,H,W) → (B,3,H,W)
        x3 = x.repeat(1, 3, 1, 1)
        feat = self.backbone.forward_features(x3)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        out = self.head(feat)
        # 入力サイズに揃える
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


class SmallUNetLandmark(nn.Module):
    """
    軽量 UNet ベースの Stage 2 モデル（GPU非推奨環境でのテスト用）。
    HRNetLandmark と同じ入出力インターフェース。
    """

    def __init__(self, num_channels: int = 4):
        super().__init__()
        from train.model import SmallUNet
        # SmallUNet は 1ch 入力をそのまま受け付ける
        self._unet = SmallUNet(num_landmarks=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._unet(x)


def get_stage2_model(backbone: str = "hrnet_w32", num_channels: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Factory 関数。
      backbone="hrnet_w32" → HRNetLandmark
      backbone="smallunet" → SmallUNetLandmark（テスト用）
    """
    if backbone == "smallunet":
        return SmallUNetLandmark(num_channels=num_channels)
    if backbone == "hrnet_w32":
        return HRNetLandmark(num_channels=num_channels, pretrained=pretrained)
    raise ValueError(f"Unknown backbone: {backbone}. Choose 'hrnet_w32' or 'smallunet'.")
