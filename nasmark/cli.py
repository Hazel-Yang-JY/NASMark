"""Shared command-line helpers."""

from __future__ import annotations

import argparse

import torch

from .data import build_cifar10_loaders, build_synthetic_loaders
from .models.supernet import SPOSMobileNetSupernet


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default="./data")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--trigger-count", type=int, default=100)
    parser.add_argument("--watermark-scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="use tiny synthetic data")


def make_model(args) -> SPOSMobileNetSupernet:
    if args.smoke_test:
        return SPOSMobileNetSupernet(
            channels=(4, 4, 8, 8),
            downsample_layers=(1,),
            watermark_layers=(1, 2),
            watermark_channels=4,
            watermark_scale=args.watermark_scale,
        )
    return SPOSMobileNetSupernet(watermark_scale=args.watermark_scale)


def make_loaders(args):
    if args.smoke_test:
        return build_synthetic_loaders(batch_size=2, samples=4, target_class=args.target_class)
    return build_cifar10_loaders(
        args.data,
        args.batch_size,
        args.workers,
        args.trigger_count,
        args.target_class,
        args.seed,
        args.download,
    )

