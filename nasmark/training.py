"""Stage 1 supernet training and Stage 2 NASMark watermark embedding."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

import torch
from torch import nn

from .models.supernet import SPOSMobileNetSupernet


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_fixed_choices(
    model: SPOSMobileNetSupernet, count: int = 4, seed: int = 0
) -> tuple[tuple[int, ...], ...]:
    """Create deterministic paths used for comparable epoch-to-epoch metrics."""
    if count < 1:
        raise ValueError("count must be positive")
    anchors = [
        tuple([candidate] * model.layers)
        for candidate in range(min(model.num_choices, count))
    ]
    rng = random.Random(seed)
    while len(anchors) < count:
        candidate = model.sample_choice(rng)
        if candidate not in anchors:
            anchors.append(candidate)
    return tuple(anchors[:count])


def train_stage1_epoch(
    model: SPOSMobileNetSupernet,
    loader,
    optimizer,
    device,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Train theta, psi, and phi together on clean main-task data."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    rng = rng or random.Random()
    total_loss = total_correct = total_examples = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        choice = model.sample_choice(rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, choice)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch = targets.numel()
        total_loss += loss.item() * batch
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_examples += batch
    return _metrics(total_loss, total_correct, total_examples)


def _searchable_weight_parameters(model: SPOSMobileNetSupernet):
    modules = dict(model.named_modules())
    for name, parameter in model.named_parameters():
        if not name.startswith("candidate_blocks.") or not name.endswith(".weight"):
            continue
        module_name = name.rsplit(".", 1)[0]
        if isinstance(modules[module_name], nn.Conv2d):
            yield name, parameter


def estimate_contribution_scores(
    model: SPOSMobileNetSupernet,
    loader,
    device,
    num_batches: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate |<w, dL/dw>| for searchable convolution weight tensors."""
    if num_batches < 1:
        raise ValueError("num_batches must be positive")
    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss()
    rng = random.Random(seed)
    scores = {name: 0.0 for name, _ in _searchable_weight_parameters(model)}
    seen = {name: 0 for name in scores}
    parameters = dict(model.named_parameters())

    for batch_index, (images, targets) in enumerate(loader):
        if batch_index >= num_batches:
            break
        images, targets = images.to(device), targets.to(device)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(images, model.sample_choice(rng)), targets)
        loss.backward()
        for name in scores:
            parameter = parameters[name]
            if parameter.grad is not None:
                scores[name] += torch.sum(parameter.detach() * parameter.grad.detach()).abs().item()
                seen[name] += 1

    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return {name: scores[name] / seen[name] for name in scores if seen[name]}


def select_low_contribution(scores: dict[str, float], rho: float) -> tuple[str, ...]:
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    if not scores:
        raise ValueError("no contribution scores were collected")
    count = max(1, math.ceil(len(scores) * rho))
    return tuple(name for name, _ in sorted(scores.items(), key=lambda item: item[1])[:count])


def configure_stage2_parameters(
    model: SPOSMobileNetSupernet, selected_backbone_parameters: Iterable[str]
) -> list[nn.Parameter]:
    """Freeze theta except theta_cpl; update theta_cpl, fixed shared part psi, and phi."""
    selected = set(selected_backbone_parameters)
    known = dict(model.named_parameters())
    missing = selected.difference(known)
    if missing:
        raise ValueError(f"unknown parameters selected: {sorted(missing)}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name in selected:
        known[name].requires_grad_(True)
    for parameter in model.shared_parameters():
        parameter.requires_grad_(True)
    for parameter in model.watermark_modules.parameters():
        parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _keep_frozen_backbone_bn_fixed(model: SPOSMobileNetSupernet) -> None:
    """Prevent frozen searchable BN layers from changing running statistics."""
    for module in model.candidate_blocks.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def _keep_stem_bn_statistics_fixed(model: SPOSMobileNetSupernet) -> None:
    """Keep the Stage 1 stem statistics while still optimizing affine/conv weights."""
    for module in model.stem.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def train_stage2_epoch(
    model: SPOSMobileNetSupernet,
    clean_loader,
    trigger_loader,
    optimizer,
    device,
    watermark_weight: float = 1.0,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Jointly optimize L_main + lambda_wm L_wm with one sampled path per step."""
    model.train()
    _keep_frozen_backbone_bn_fixed(model)
    _keep_stem_bn_statistics_fixed(model)
    criterion = nn.CrossEntropyLoss()
    rng = rng or random.Random()
    trigger_iterator = iter(trigger_loader)
    total_loss = total_clean_correct = total_wm_correct = total_examples = total_wm_examples = 0

    for clean_images, clean_targets in clean_loader:
        try:
            trigger_images, trigger_targets = next(trigger_iterator)
        except StopIteration:
            trigger_iterator = iter(trigger_loader)
            trigger_images, trigger_targets = next(trigger_iterator)
        clean_images, clean_targets = clean_images.to(device), clean_targets.to(device)
        trigger_images, trigger_targets = trigger_images.to(device), trigger_targets.to(device)
        choice = model.sample_choice(rng)

        optimizer.zero_grad(set_to_none=True)
        # One mixed forward keeps BatchNorm statistics proportional to the
        # actual  clean:trigger sample ratio instead of updating them once per
        # loader regardless of batch size.
        clean_batch = clean_targets.numel()
        combined_images = torch.cat((clean_images, trigger_images), dim=0)
        combined_logits = model(combined_images, choice)
        clean_logits = combined_logits[:clean_batch]
        trigger_logits = combined_logits[clean_batch:]
        clean_loss = criterion(clean_logits, clean_targets)
        watermark_loss = criterion(trigger_logits, trigger_targets)
        loss = clean_loss + watermark_weight * watermark_loss
        loss.backward()
        optimizer.step()

        batch = clean_batch
        total_loss += loss.item() * batch
        total_clean_correct += (clean_logits.argmax(1) == clean_targets).sum().item()
        total_wm_correct += (trigger_logits.argmax(1) == trigger_targets).sum().item()
        total_examples += batch
        total_wm_examples += trigger_targets.numel()

    metrics = _metrics(total_loss, total_clean_correct, total_examples)
    metrics["watermark_accuracy"] = 100.0 * total_wm_correct / max(total_wm_examples, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, choice=None, target_class: int | None = None) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = total_correct = total_target_predictions = total_examples = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images, choice) if isinstance(model, SPOSMobileNetSupernet) else model(images)
        loss = criterion(logits, targets)
        batch = targets.numel()
        total_loss += loss.item() * batch
        total_correct += (logits.argmax(1) == targets).sum().item()
        if target_class is not None:
            total_target_predictions += (logits.argmax(1) == target_class).sum().item()
        total_examples += batch
    metrics = _metrics(total_loss, total_correct, total_examples)
    if target_class is not None:
        metrics["target_prediction_rate"] = 100.0 * total_target_predictions / max(total_examples, 1)
    return metrics


def evaluate_paths(
    model: SPOSMobileNetSupernet,
    loader,
    device,
    choices,
    target_class: int | None = None,
) -> dict[str, float]:
    """Average metrics over fixed paths to avoid judging a supernet by one draw."""
    path_metrics = [evaluate(model, loader, device, choice, target_class) for choice in choices]
    keys = path_metrics[0].keys()
    result = {key: sum(item[key] for item in path_metrics) / len(path_metrics) for key in keys}
    result["min_accuracy"] = min(item["accuracy"] for item in path_metrics)
    result["max_accuracy"] = max(item["accuracy"] for item in path_metrics)
    return result


def _metrics(total_loss: float, total_correct: int, total_examples: int) -> dict[str, float]:
    denominator = max(total_examples, 1)
    return {
        "loss": total_loss / denominator,
        "accuracy": 100.0 * total_correct / denominator,
    }
