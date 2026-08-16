"""SPOS MobileNet supernet with layer-shared NASMark residual branches."""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence

import torch
from torch import nn

from .blocks import MobileInvertedResidual, WatermarkModule


DEFAULT_CHANNELS = (16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320)
DEFAULT_EXPAND_RATIOS = (1, 3, 6, 4)
DEFAULT_DOWNSAMPLE_LAYERS = (1, 3, 6)
DEFAULT_WATERMARK_LAYERS = (7, 8, 9)


def _validate_choice(choice: Sequence[int], layers: int, num_choices: int) -> tuple[int, ...]:
    if len(choice) != layers:
        raise ValueError(f"choice must contain {layers} entries, got {len(choice)}")
    result = tuple(int(item) for item in choice)
    if any(item < 0 or item >= num_choices for item in result):
        raise ValueError(f"choice entries must be in [0, {num_choices - 1}]")
    return result


class SPOSMobileNetSupernet(nn.Module):
    """A single-path one-shot MobileNet supernet for CIFAR-10.

    There is exactly one watermark module at each selected layer. It receives
    the layer input and is added to whichever candidate path SPOS samples.
    """

    def __init__(
        self,
        num_classes: int = 10,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        expand_ratios: Sequence[int] = DEFAULT_EXPAND_RATIOS,
        downsample_layers: Sequence[int] = DEFAULT_DOWNSAMPLE_LAYERS,
        watermark_layers: Sequence[int] = DEFAULT_WATERMARK_LAYERS,
        watermark_channels: int = 32,
        watermark_scale: float = 0.2,
        watermark_final_relu: bool = True,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must describe a stem and at least one searchable layer")
        self.channels = tuple(int(c) for c in channels)
        self.layers = len(self.channels) - 1
        self.expand_ratios = tuple(int(r) for r in expand_ratios)
        self.num_choices = len(self.expand_ratios)
        self.downsample_layers = frozenset(int(i) for i in downsample_layers)
        self.watermark_layers = tuple(sorted(set(int(i) for i in watermark_layers)))
        self.watermark_scale = float(watermark_scale)

        invalid = [i for i in (*self.downsample_layers, *self.watermark_layers) if not 0 <= i < self.layers]
        if invalid:
            raise ValueError(f"layer indices out of range: {invalid}")

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU6(inplace=True),
        )

        self.candidate_blocks = nn.ModuleList()
        self.watermark_modules = nn.ModuleDict()
        for layer_index in range(self.layers):
            in_channels = self.channels[layer_index]
            out_channels = self.channels[layer_index + 1]
            stride = 2 if layer_index in self.downsample_layers else 1
            self.candidate_blocks.append(
                nn.ModuleList(
                    MobileInvertedResidual(in_channels, out_channels, stride, ratio)
                    for ratio in self.expand_ratios
                )
            )
            if layer_index in self.watermark_layers:
                self.watermark_modules[str(layer_index)] = WatermarkModule(
                    in_channels,
                    out_channels,
                    stride=stride,
                    bottleneck_channels=watermark_channels,
                    final_relu=watermark_final_relu,
                )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.channels[-1], num_classes)
        self.reset_parameters()

    def sample_choice(self, rng: random.Random | None = None) -> tuple[int, ...]:
        rng = rng or random
        return tuple(rng.randrange(self.num_choices) for _ in range(self.layers))

    def forward(self, x: torch.Tensor, choice: Sequence[int] | None = None) -> torch.Tensor:
        selected = self.sample_choice() if choice is None else _validate_choice(
            choice, self.layers, self.num_choices
        )
        x = self.stem(x)
        for layer_index, candidate_index in enumerate(selected):
            layer_input = x
            x = self.candidate_blocks[layer_index][candidate_index](layer_input)
            key = str(layer_index)
            if key in self.watermark_modules:
                x = x + self.watermark_scale * self.watermark_modules[key](layer_input)
        x = self.global_pooling(x).flatten(1)
        return self.classifier(x)

    def searchable_parameters(self):
        return self.candidate_blocks.parameters()

    def shared_parameters(self):
        yield from self.stem.parameters()
        yield from self.classifier.parameters()

    def export_subnet(self, choice: Sequence[int]) -> "SPOSMobileNetSubnet":
        return SPOSMobileNetSubnet(self, _validate_choice(choice, self.layers, self.num_choices))

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class SPOSMobileNetSubnet(nn.Module):
    """A standalone subnet copied from a trained NASMark supernet."""

    def __init__(self, supernet: SPOSMobileNetSupernet, choice: Sequence[int]):
        super().__init__()
        self.choice = tuple(choice)
        self.watermark_scale = supernet.watermark_scale
        self.stem = copy.deepcopy(supernet.stem)
        self.blocks = nn.ModuleList(
            copy.deepcopy(supernet.candidate_blocks[i][candidate])
            for i, candidate in enumerate(self.choice)
        )
        self.watermark_modules = copy.deepcopy(supernet.watermark_modules)
        self.global_pooling = copy.deepcopy(supernet.global_pooling)
        self.classifier = copy.deepcopy(supernet.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer_index, block in enumerate(self.blocks):
            layer_input = x
            x = block(layer_input)
            key = str(layer_index)
            if key in self.watermark_modules:
                x = x + self.watermark_scale * self.watermark_modules[key](layer_input)
        x = self.global_pooling(x).flatten(1)
        return self.classifier(x)

