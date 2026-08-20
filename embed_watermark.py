"""Stage 2: embed the trigger behavior using interdependent watermark training."""

import argparse
import random

import torch

from nasmark.checkpoint import load_checkpoint, save_checkpoint
from nasmark.cli import WATERMARK_CLASS, add_common_arguments, make_loaders, make_model
from nasmark.models.supernet import WatermarkClassHead, add_watermark_class
from nasmark.training import (
    configure_stage2_parameters,
    estimate_contribution_scores,
    evaluate_paths,
    make_fixed_choices,
    select_low_contribution,
    set_seed,
    train_stage2_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--input", default="checkpoints/stage1_supernet.pt")
    parser.add_argument("--output", default="checkpoints/nasmark_supernet.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--backbone-learning-rate", type=float, default=0.0001)
    parser.add_argument("--watermark-learning-rate", type=float, default=0.005)
    parser.add_argument("--stem-learning-rate", type=float, default=0.0001)
    parser.add_argument("--watermark-head-learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--watermark-weight", type=float, default=0.5)
    parser.add_argument("--watermark-warmup-epochs", type=int, default=5)
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--score-batches", type=int, default=100)
    parser.add_argument("--validation-paths", type=int, default=4)
    parser.add_argument("--max-clean-drop", type=float, default=3.0)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    # Stage 1 learns the ten CIFAR-10 logits. Stage 2 always appends the
    # dedicated watermark logit after loading that reusable checkpoint.
    model = make_model(args).to(device)
    load_checkpoint(args.input, model, device)
    add_watermark_class(model, freeze_base=True)
    loaders = make_loaders(args)
    validation_choices = make_fixed_choices(model, args.validation_paths, args.seed)
    baseline_clean = evaluate_paths(
        model, loaders["valid"], device, validation_choices, WATERMARK_CLASS
    )
    baseline_watermark = evaluate_paths(
        model, loaders["trigger_valid"], device, validation_choices
    )
    print(
        f"stage2 baseline: clean_mean={baseline_clean['accuracy']:.2f} "
        f"clean_range=[{baseline_clean['min_accuracy']:.2f},{baseline_clean['max_accuracy']:.2f}] "
        f"WSR={baseline_watermark['accuracy']:.2f} "
        f"clean_target_rate={baseline_clean['target_prediction_rate']:.2f}"
    )

    scores = estimate_contribution_scores(
        model, loaders["train"], device, args.score_batches, args.seed
    )
    selected = select_low_contribution(scores, args.rho)
    configure_stage2_parameters(model, selected)
    if not isinstance(model.classifier, WatermarkClassHead):
        raise TypeError("Stage 2 requires the dedicated watermark class head")
    for parameter in model.classifier.base_classifier.parameters():
        parameter.requires_grad_(False)
    named_parameters = dict(model.named_parameters())
    parameter_groups = [
        {
            "params": [named_parameters[name] for name in selected],
            "lr": args.backbone_learning_rate,
            "name": "low_contribution_backbone",
        },
        {
            "params": list(model.watermark_modules.parameters()),
            "lr": args.watermark_learning_rate,
            "name": "watermark_modules",
        },
        {
            "params": list(model.stem.parameters()),
            "lr": args.stem_learning_rate,
            "name": "shared_stem",
        },
    ]
    parameter_groups.append(
        {
            "params": list(model.classifier.watermark_classifier.parameters()),
            "lr": args.watermark_head_learning_rate,
            "name": "watermark_class_head",
        }
    )
    optimizer = torch.optim.SGD(
        parameter_groups,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    rng = random.Random(args.seed)
    print(f"selected {len(selected)}/{len(scores)} low-contribution backbone tensors")
    print(
        "stage2 learning rates: "
        f"backbone={args.backbone_learning_rate:g}, watermark={args.watermark_learning_rate:g}, "
        f"stem={args.stem_learning_rate:g}, "
        f"watermark_head={args.watermark_head_learning_rate:g}, base_classifier=frozen"
    )

    best_watermark_accuracy = baseline_watermark["accuracy"]
    best_clean_accuracy = baseline_clean["accuracy"]
    best_epoch = 0
    clean_drop_violations = 0
    save_checkpoint(
        args.output,
        model,
        stage="stage2",
        epoch=0,
        baseline_clean_accuracy=baseline_clean["accuracy"],
        clean_accuracy=baseline_clean["accuracy"],
        watermark_accuracy=baseline_watermark["accuracy"],
        watermark_class=WATERMARK_CLASS,
        trigger_patch_size=args.trigger_patch_size,
        selected_backbone_parameters=selected,
        validation_choices=validation_choices,
        rho=args.rho,
        watermark_scale=args.watermark_scale,
        smoke_test=args.smoke_test,
    )

    for epoch in range(args.epochs):
        warmup_fraction = min(1.0, (epoch + 1) / max(1, args.watermark_warmup_epochs))
        current_watermark_weight = args.watermark_weight * warmup_fraction
        metrics = train_stage2_epoch(
            model,
            loaders["train"],
            loaders["trigger_train"],
            optimizer,
            device,
            current_watermark_weight,
            rng,
        )
        clean = evaluate_paths(
            model, loaders["valid"], device, validation_choices, WATERMARK_CLASS
        )
        watermark = evaluate_paths(
            model, loaders["trigger_valid"], device, validation_choices
        )
        clean_drop = baseline_clean["accuracy"] - clean["accuracy"]
        eligible = clean_drop <= args.max_clean_drop
        if eligible:
            clean_drop_violations = 0
            improved = watermark["accuracy"] > best_watermark_accuracy or (
                watermark["accuracy"] == best_watermark_accuracy
                and clean["accuracy"] > best_clean_accuracy
            )
            if improved:
                best_watermark_accuracy = watermark["accuracy"]
                best_clean_accuracy = clean["accuracy"]
                best_epoch = epoch + 1
                save_checkpoint(
                    args.output,
                    model,
                    stage="stage2",
                    epoch=epoch + 1,
                    baseline_clean_accuracy=baseline_clean["accuracy"],
                    clean_accuracy=clean["accuracy"],
                    watermark_accuracy=watermark["accuracy"],
                    watermark_class=WATERMARK_CLASS,
                    trigger_patch_size=args.trigger_patch_size,
                    clean_target_prediction_rate=clean["target_prediction_rate"],
                    selected_backbone_parameters=selected,
                    validation_choices=validation_choices,
                    rho=args.rho,
                    watermark_weight=args.watermark_weight,
                    watermark_scale=args.watermark_scale,
                    smoke_test=args.smoke_test,
                )
        else:
            clean_drop_violations += 1
        scheduler.step()
        print(
            f"stage2 epoch={epoch + 1}/{args.epochs} lambda={current_watermark_weight:.4f} "
            f"loss={metrics['loss']:.4f} clean_mean={clean['accuracy']:.2f} "
            f"clean_drop={clean_drop:.2f} WSR_mean={watermark['accuracy']:.2f} "
            f"target_rate={clean['target_prediction_rate']:.2f} "
            f"best_epoch={best_epoch} best_WSR={best_watermark_accuracy:.2f}"
        )
        if clean_drop_violations >= args.early_stop_patience:
            print(
                f"early stop: clean accuracy exceeded the {args.max_clean_drop:.2f}-point "
                f"drop limit for {clean_drop_violations} consecutive epochs"
            )
            break

    print(
        f"saved best eligible watermarked supernet: {args.output} "
        f"(epoch={best_epoch}, clean={best_clean_accuracy:.2f}, "
        f"WSR={best_watermark_accuracy:.2f})"
    )


if __name__ == "__main__":
    main()
