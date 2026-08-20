"""Training-free architecture search and standalone subnet extraction."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .models.supernet import SPOSMobileNetSupernet
from .training import evaluate


@dataclass(frozen=True)
class SearchResult:
    choice: tuple[int, ...]
    validation_accuracy: float
    watermark_success_rate: float
    parameters: int


def count_subnet_parameters(model: SPOSMobileNetSupernet, choice) -> int:
    if len(choice) != model.layers:
        raise ValueError(f"choice must contain {model.layers} entries")
    fixed = sum(parameter.numel() for parameter in model.stem.parameters())
    fixed += sum(parameter.numel() for parameter in model.watermark_modules.parameters())
    fixed += sum(parameter.numel() for parameter in model.classifier.parameters())
    selected = sum(
        parameter.numel()
        for layer_index, candidate_index in enumerate(choice)
        for parameter in model.candidate_blocks[layer_index][candidate_index].parameters()
    )
    return fixed + selected


def random_search(
    model: SPOSMobileNetSupernet,
    validation_loader,
    trigger_loader,
    device,
    samples: int,
    max_parameters: int | None = None,
    min_parameters: int = 0,
    seed: int = 0,
) -> list[SearchResult]:
    """Evaluate frozen candidates; this function never trains or recovers weights."""
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    results: list[SearchResult] = []
    seen: set[tuple[int, ...]] = set()
    maximum_unique = model.num_choices**model.layers
    attempts = 0
    max_attempts = max(samples * 100, 1000)

    model.eval()
    while len(results) < samples and len(seen) < maximum_unique and attempts < max_attempts:
        attempts += 1
        choice = model.sample_choice(rng)
        if choice in seen:
            continue
        seen.add(choice)
        parameter_count = count_subnet_parameters(model, choice)
        if parameter_count < min_parameters:
            continue
        if max_parameters is not None and parameter_count > max_parameters:
            continue
        clean = evaluate(model, validation_loader, device, choice)
        watermark = evaluate(model, trigger_loader, device, choice)
        results.append(
            SearchResult(choice, clean["accuracy"], watermark["accuracy"], parameter_count)
        )
    if len(results) < samples:
        raise RuntimeError(
            f"only {len(results)} architectures satisfy the resource constraint after {attempts} attempts"
        )
    return sorted(results, key=lambda result: result.validation_accuracy, reverse=True)


def save_subnet(path, supernet: SPOSMobileNetSupernet, result: SearchResult) -> None:
    subnet = supernet.export_subnet(result.choice)
    payload = {
        "choice": result.choice,
        "validation_accuracy": result.validation_accuracy,
        "watermark_success_rate": result.watermark_success_rate,
        "parameters": result.parameters,
        "model": subnet.state_dict(),
    }
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
