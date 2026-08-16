"""Search the frozen NASMark supernet and export the best SPOS subnet."""

import argparse

import torch

from nasmark.checkpoint import load_checkpoint
from nasmark.cli import add_common_arguments, make_loaders, make_model
from nasmark.search import random_search, save_subnet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--input", default="checkpoints/nasmark_supernet.pt")
    parser.add_argument("--output", default="checkpoints/best_subnet.pt")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--min-parameters", type=int, default=0)
    parser.add_argument("--max-parameters", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = make_model(args).to(device)
    load_checkpoint(args.input, model, device)
    loaders = make_loaders(args)

    results = random_search(
        model,
        loaders["valid"],
        loaders["trigger_valid"],
        device,
        args.samples,
        args.max_parameters,
        args.min_parameters,
        args.seed,
    )
    for rank, result in enumerate(results[:10], 1):
        print(
            f"rank={rank} choice={list(result.choice)} params={result.parameters} "
            f"val_acc={result.validation_accuracy:.2f} WSR={result.watermark_success_rate:.2f}"
        )
    save_subnet(args.output, model, results[0])
    print(f"saved best frozen subnet without recover/retraining: {args.output}")


if __name__ == "__main__":
    main()

