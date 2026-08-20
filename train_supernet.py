"""Stage 1: train the SPOS watermark supernet on clean CIFAR-10."""

import argparse
import random
from pathlib import Path

import torch

from nasmark.checkpoint import load_checkpoint, save_checkpoint
from nasmark.cli import add_common_arguments, make_loaders, make_model
from nasmark.training import evaluate_paths, make_fixed_choices, set_seed, train_stage1_epoch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--validation-paths", type=int, default=4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--output", default="checkpoints/stage1_supernet.pt")
    parser.add_argument("--last-output", help="last-state checkpoint used for resuming")
    parser.add_argument("--resume", help="resume from a last-state checkpoint")
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
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    rng = random.Random(args.seed)
    validation_choices = make_fixed_choices(model, args.validation_paths, args.seed)
    best_validation_accuracy = float("-inf")
    start_epoch = 0
    last_output = args.last_output or str(
        Path(args.output).with_name(f"{Path(args.output).stem}_last{Path(args.output).suffix}")
    )

    if args.resume:
        state = load_checkpoint(args.resume, model, device)
        if state.get("stage") != "stage1-last":
            raise ValueError("--resume must point to a Stage 1 last-state checkpoint")
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"])
        best_validation_accuracy = float(state["best_validation_accuracy"])
        validation_choices = tuple(tuple(choice) for choice in state["validation_choices"])
        rng.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        print(
            f"resumed Stage 1 from epoch {start_epoch}; "
            f"best validation accuracy={best_validation_accuracy:.2f}"
        )

    for epoch in range(start_epoch, args.epochs):
        if epoch < args.warmup_epochs:
            warmup_lr = args.learning_rate * (epoch + 1) / args.warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        current_learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_stage1_epoch(model, loaders["train"], optimizer, device, rng)
        validation = evaluate_paths(model, loaders["valid"], device, validation_choices)
        if epoch >= args.warmup_epochs - 1:
            scheduler.step()
        if validation["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation["accuracy"]
            save_checkpoint(
                args.output,
                model,
                stage="stage1",
                epoch=epoch + 1,
                validation_accuracy=validation["accuracy"],
                validation_choices=validation_choices,
                watermark_scale=args.watermark_scale,
                smoke_test=args.smoke_test,
            )
        print(
            f"stage1 epoch={epoch + 1}/{args.epochs} "
            f"lr={current_learning_rate:.6f} loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.2f} val_mean={validation['accuracy']:.2f} "
            f"val_range=[{validation['min_accuracy']:.2f},{validation['max_accuracy']:.2f}] "
            f"best={best_validation_accuracy:.2f}"
        )
        save_checkpoint(
            last_output,
            model,
            stage="stage1-last",
            epoch=epoch + 1,
            best_validation_accuracy=best_validation_accuracy,
            validation_choices=validation_choices,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            python_rng_state=rng.getstate(),
            torch_rng_state=torch.get_rng_state(),
            watermark_scale=args.watermark_scale,
            smoke_test=args.smoke_test,
        )

    print(
        f"saved best Stage 1 watermark supernet: {args.output} "
        f"(mean validation accuracy={best_validation_accuracy:.2f}); "
        f"last state: {last_output}"
    )


if __name__ == "__main__":
    main()
