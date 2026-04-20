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
