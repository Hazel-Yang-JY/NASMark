"""Report clean accuracy and watermark success rate for a fixed supernet path."""

import argparse

import torch

from nasmark.checkpoint import load_checkpoint
from nasmark.cli import add_common_arguments, make_loaders, make_model
from nasmark.training import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--input", default="checkpoints/nasmark_supernet.pt")
    parser.add_argument("--choice", required=True, help="comma-separated candidate indices")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = make_model(args).to(device)
    load_checkpoint(args.input, model, device)
    choice = tuple(int(item) for item in args.choice.split(","))
    loaders = make_loaders(args)
    clean = evaluate(model, loaders["valid"], device, choice)
    watermark = evaluate(model, loaders["trigger_valid"], device, choice)
    print(f"clean_accuracy={clean['accuracy']:.2f} watermark_success_rate={watermark['accuracy']:.2f}")


if __name__ == "__main__":
    main()

