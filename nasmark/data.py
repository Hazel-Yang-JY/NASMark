"""CIFAR-10 and deterministic patch-trigger data helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
WATERMARK_CLASS = 10


class PatchTriggerDataset(Dataset):
    """A fixed subset with a white square trigger and one target class."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        patch_size: int = 3,
        margin: int = 1,
    ):
        self.base_dataset = base_dataset
        self.indices = tuple(int(i) for i in indices)
        self.patch_size = int(patch_size)
        self.margin = int(margin)
        self.white = torch.tensor(
            [(1.0 - mean) / std for mean, std in zip(CIFAR10_MEAN, CIFAR10_STD)]
        ).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, _ = self.base_dataset[self.indices[index]]
        if not isinstance(image, torch.Tensor):
            raise TypeError("base_dataset must return transformed tensors")
        image = image.clone()
        end_h = image.shape[-2] - self.margin
        end_w = image.shape[-1] - self.margin
        start_h = end_h - self.patch_size
        start_w = end_w - self.patch_size
        image[:, start_h:end_h, start_w:end_w] = self.white.to(image.dtype)
        return image, WATERMARK_CLASS


def cifar10_transforms(train: bool):
    operations: list[object] = []
    if train:
        operations.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    operations.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return transforms.Compose(operations)


def choose_trigger_indices(dataset, count: int, seed: int) -> list[int]:
    candidates = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if count > len(candidates):
        raise ValueError(f"requested {count} triggers from only {len(candidates)} candidates")
    return candidates[:count]


def build_cifar10_loaders(
    root: str,
    batch_size: int,
    workers: int,
    trigger_count: int,
    seed: int,
    download: bool = False,
    trigger_batch_size: int | None = None,
    trigger_patch_size: int = 3,
):
    train_set = datasets.CIFAR10(root, train=True, download=download, transform=cifar10_transforms(True))
    train_eval_set = datasets.CIFAR10(
        root, train=True, download=False, transform=cifar10_transforms(False)
    )
    valid_set = datasets.CIFAR10(root, train=False, download=download, transform=cifar10_transforms(False))
    train_indices = choose_trigger_indices(train_set, trigger_count, seed)
    valid_indices = choose_trigger_indices(valid_set, trigger_count, seed + 1)
    trigger_train = PatchTriggerDataset(
        train_eval_set, train_indices, patch_size=trigger_patch_size
    )
    trigger_valid = PatchTriggerDataset(
        valid_set, valid_indices, patch_size=trigger_patch_size
    )
    common = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    trigger_common = {
        "batch_size": trigger_batch_size or batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        "train": DataLoader(train_set, shuffle=True, **common),
        "valid": DataLoader(valid_set, shuffle=False, **common),
        "trigger_train": DataLoader(trigger_train, shuffle=True, **trigger_common),
        "trigger_valid": DataLoader(trigger_valid, shuffle=False, **trigger_common),
    }


def build_synthetic_loaders(
    batch_size: int = 4,
    samples: int = 8,
    trigger_batch_size: int | None = None,
    trigger_patch_size: int = 3,
):
    """Tiny in-memory data used only for installation smoke tests."""
    generator = torch.Generator().manual_seed(0)
    clean_x = torch.randn(samples, 3, 32, 32, generator=generator)
    clean_y = torch.randint(0, 10, (samples,), generator=generator)
    trigger_x = clean_x.clone()
    end = -1
    start = -(trigger_patch_size + 1)
    trigger_x[:, :, start:end, start:end] = 2.0
    trigger_y = torch.full((samples,), WATERMARK_CLASS, dtype=torch.long)
    return {
        "train": DataLoader(TensorDataset(clean_x, clean_y), batch_size=batch_size),
        "valid": DataLoader(TensorDataset(clean_x, clean_y), batch_size=batch_size),
        "trigger_train": DataLoader(
            TensorDataset(trigger_x, trigger_y), batch_size=trigger_batch_size or batch_size
        ),
        "trigger_valid": DataLoader(
            TensorDataset(trigger_x, trigger_y), batch_size=trigger_batch_size or batch_size
        ),
    }
