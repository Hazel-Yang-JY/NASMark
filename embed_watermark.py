"""Stage 2: embed the trigger behavior using interdependent watermark training."""

import argparse
import random

import torch

from nasmark.checkpoint import load_checkpoint, save_checkpoint
from nasmark.cli import add_common_arguments, make_loaders, make_model
from nasmark.training import (
    configure_stage2_parameters,
    estimate_contribution_scores,
    evaluate,
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
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--stem-lr-scale",
        type=float,
        default=0.05,
        help="learning-rate multiplier for the shared stem",
    )
    parser.add_argument(
        "--classifier-lr-scale",
        type=float,
        default=0.1,
        help="learning-rate multiplier for the shared classifier",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--watermark-weight", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--score-batches", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    model = make_model(args).to(device)
    load_checkpoint(args.input, model, device)
    loaders = make_loaders(args)

    scores = estimate_contribution_scores(
        model, loaders["train"], device, args.score_batches, args.seed
    )
    selected = select_low_contribution(scores, args.rho)
    configure_stage2_parameters(model, selected)
    named_parameters = dict(model.named_parameters())
    parameter_groups = [
        {
            "params": [named_parameters[name] for name in selected],
            "lr": args.learning_rate,
            "name": "low_contribution_backbone",
        },
        {
            "params": list(model.watermark_modules.parameters()),
            "lr": args.learning_rate,
            "name": "watermark_modules",
        },
        {
            "params": list(model.stem.parameters()),
            "lr": args.learning_rate * args.stem_lr_scale,
            "name": "shared_stem",
        },
        {
            "params": list(model.classifier.parameters()),
            "lr": args.learning_rate * args.classifier_lr_scale,
            "name": "shared_classifier",
        },
    ]
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
        f"backbone={args.learning_rate:g}, watermark={args.learning_rate:g}, "
        f"stem={args.learning_rate * args.stem_lr_scale:g}, "
        f"classifier={args.learning_rate * args.classifier_lr_scale:g}"
    )

    for epoch in range(args.epochs):
        metrics = train_stage2_epoch(
            model,
            loaders["train"],
            loaders["trigger_train"],
            optimizer,
            device,
            args.watermark_weight,
            rng,
        )
        choice = model.sample_choice(rng)
        clean = evaluate(model, loaders["valid"], device, choice)
        watermark = evaluate(model, loaders["trigger_valid"], device, choice)
        scheduler.step()
        print(
            f"stage2 epoch={epoch + 1}/{args.epochs} loss={metrics['loss']:.4f} "
            f"clean_acc={clean['accuracy']:.2f} WSR={watermark['accuracy']:.2f}"
        )

    save_checkpoint(
        args.output,
        model,
        stage="stage2",
        rho=args.rho,
        selected_backbone_parameters=selected,
        watermark_scale=args.watermark_scale,
        smoke_test=args.smoke_test,
    )
    print(f"saved watermarked supernet: {args.output}")


if __name__ == "__main__":
    main()
