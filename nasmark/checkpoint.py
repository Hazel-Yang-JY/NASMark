"""Portable checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, **metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **metadata}, path)


def load_checkpoint(path, model, device="cpu") -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return {key: value for key, value in checkpoint.items() if key != "model"}

