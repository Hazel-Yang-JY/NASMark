"""Building blocks for the SPOS MobileNet search space and NASMark branches."""

from __future__ import annotations

import torch
from torch import nn


class MobileInvertedResidual(nn.Module):
    """MobileNetV2-style candidate: 1x1 expansion, 3x3 DW, 1x1 projection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden_channels = in_channels * expand_ratio
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_residual else out


class WatermarkModule(nn.Module):
    """A lightweight residual branch shared by every candidate at one layer.

    The branch follows the paper's convolutional bottleneck:
    1x1 projection -> 3x3 depthwise convolution -> 1x1 projection.
    The final ReLU is retained from the author's existing implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bottleneck_channels: int = 32,
        final_relu: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                3,
                stride=stride,
                padding=1,
                groups=bottleneck_channels,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if final_relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

