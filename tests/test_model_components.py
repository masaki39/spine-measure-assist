import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train")))

import torch
import pytest

from model import ConvBlock, SmallUNet, UpBlock


def test_conv_block_shape():
    block = ConvBlock(16, 32)
    x = torch.ones(1, 16, 64, 64)
    out = block(x)
    assert out.shape == (1, 32, 64, 64)


def test_up_block_shape():
    block = UpBlock(in_ch=64, skip_ch=32, out_ch=32)
    x = torch.ones(1, 64, 16, 16)
    skip = torch.ones(1, 32, 32, 32)
    out = block(x, skip)
    assert out.shape == (1, 32, 32, 32)


def test_up_block_odd_dimensions():
    block = UpBlock(in_ch=64, skip_ch=32, out_ch=32)
    x = torch.ones(1, 64, 15, 15)
    skip = torch.ones(1, 32, 31, 31)
    out = block(x, skip)
    assert out.shape == (1, 32, 31, 31)


def test_small_unet_output_channels():
    model = SmallUNet(num_landmarks=3)
    x = torch.ones(1, 1, 128, 96)
    out = model(x)
    assert out.shape[1] == 3


def test_small_unet_output_spatial():
    model = SmallUNet(num_landmarks=5)
    x = torch.ones(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 5, 128, 128)


def test_small_unet_gradient_flow():
    model = SmallUNet(num_landmarks=5)
    x = torch.ones(1, 1, 64, 64)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"


# --- get_model factory ---

from model import get_model


def test_get_model_smallunet_returns_small_unet():
    model = get_model("smallunet", num_landmarks=6)
    assert isinstance(model, SmallUNet)


def test_get_model_smallunet_correct_output_channels():
    model = get_model("smallunet", num_landmarks=6)
    x = torch.ones(1, 1, 128, 128)
    out = model(x)
    assert out.shape[1] == 6


def test_get_model_smallunet_pretrained_flag_ignored():
    # SmallUNet has no pretrained weights — flag should be silently ignored
    model = get_model("smallunet", num_landmarks=5, pretrained=True)
    assert isinstance(model, SmallUNet)


def test_get_model_unknown_backbone_raises():
    with pytest.raises((ImportError, Exception)):
        get_model("nonexistent_backbone_xyz", num_landmarks=5, pretrained=False)


# --- ConvBlock additional ---

def test_conv_block_batch_size_two():
    block = ConvBlock(8, 16)
    x = torch.ones(2, 8, 32, 32)
    out = block(x)
    assert out.shape == (2, 16, 32, 32)


def test_conv_block_preserves_spatial_dims():
    block = ConvBlock(4, 8)
    x = torch.ones(1, 4, 47, 53)  # odd non-square dimensions
    out = block(x)
    assert out.shape[2:] == (47, 53)


# --- UpBlock additional ---

def test_up_block_various_channel_sizes():
    block = UpBlock(in_ch=128, skip_ch=64, out_ch=64)
    x = torch.ones(1, 128, 8, 8)
    skip = torch.ones(1, 64, 16, 16)
    out = block(x, skip)
    assert out.shape == (1, 64, 16, 16)


# --- SmallUNet additional ---

def test_small_unet_single_landmark():
    model = SmallUNet(num_landmarks=1)
    x = torch.ones(1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_small_unet_preserves_non_square_spatial():
    model = SmallUNet(num_landmarks=6)
    x = torch.ones(1, 1, 128, 96)
    out = model(x)
    assert out.shape == (1, 6, 128, 96)


def test_small_unet_eval_mode_no_error():
    model = SmallUNet(num_landmarks=5)
    model.eval()
    with torch.no_grad():
        x = torch.ones(1, 1, 64, 64)
        out = model(x)
    assert out.shape == (1, 5, 64, 64)
