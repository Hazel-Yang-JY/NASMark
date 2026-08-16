"""Stage 1: train the SPOS watermark supernet on clean CIFAR-10."""

import argparse
import random

import torch

from nasmark.checkpoint import save_checkpoint
from nasmark.cli import add_common_arguments, make_loaders, make_model
from nasmark.training import evaluate, set_seed, train_stage1_epoch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--output", default="checkpoints/stage1_supernet.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    model = make_model(args).to(device)
    loaders = make_loaders(args)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    rng = random.Random(args.seed)

    for epoch in range(args.epochs):
        train_metrics = train_stage1_epoch(model, loaders["train"], optimizer, device, rng)
        validation = evaluate(model, loaders["valid"], device, model.sample_choice(rng))
        scheduler.step()
        print(
            f"stage1 epoch={epoch + 1}/{args.epochs} "
            f"loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.2f} "
            f"val_acc={validation['accuracy']:.2f}"
        )

    save_checkpoint(
        args.output,
        model,
        stage="stage1",
        watermark_scale=args.watermark_scale,
        smoke_test=args.smoke_test,
    )
    print(f"saved Stage 1 watermark supernet: {args.output}")


if __name__ == "__main__":
    main()
