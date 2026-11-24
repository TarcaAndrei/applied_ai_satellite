"""CNN models for Sentinel-2 patches.

    from sat.models import resnet18
    model = resnet18(in_channels=12)        # 12 L2A bands, 64x64 patches -> 10 class logits

The network is torchvision's ResNet-18, randomly initialised (no pretrained weights), with the
stem adapted to small multispectral patches:
  * conv1: 3x3, stride 1, `in_channels` inputs (torchvision: 7x7, stride 2, 3 inputs);
  * no max-pool after conv1.
The standard stem shrinks the input 4x before the first residual stage, which suits 224 px photos
but leaves a 2x2 map at the end for a 64 px patch. With this stem the stages see
64 -> 32 -> 16 -> 8 px, then global average pooling.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision


def resnet18(in_channels: int = 12, n_classes: int = 10) -> nn.Module:
    m = torchvision.models.resnet18(weights=None, num_classes=n_classes)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")  # as torchvision inits its convs
    m.maxpool = nn.Identity()
    return m


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
