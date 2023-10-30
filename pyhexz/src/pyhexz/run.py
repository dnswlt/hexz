"""Submodule to run commands using
    python -m pyhexz.run <args>
"""

import argparse
import numpy as np
import torch

from pyhexz import hexz


def parse_args():
    parser = argparse.ArgumentParser(description="Hexz NeuralMCTS")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print verbose output.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Mode to execute: selfplay to generate examples, generate to generate a new randomly initialized model, train to ... train",
        choices=("selfplay", "train", "generate", "export", "print", "hello"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device to use",
        choices=("cpu", "cuda", "mps"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        help="Path to the model to use. If this is a directory, picks the latest model checkpoint.",
    )
    parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Path or glob to the examples to use during training. Default: '{--model}/examples/*.h5'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory into which outputs are written.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=1000,
        help="Maximum number of games to play during self-play.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=60,
        help="Maximum number of seconds to play during self-play.",
    )
    parser.add_argument(
        "--runs-per-move", type=int, default=800, help="Number of MCTS runs per move."
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size to use during training."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to shuffle the examples during training.",
    )
    parser.add_argument(
        "--in-mem",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to load all HDF5 example data into memory for training.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Determines the pin_memory value used in the PyTorch DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Determines the num_workers value used in the PyTorch DataLoader. Set to 0 to disable multi-processing.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, existing outputs may be overriden.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        print(f"cuda available: {torch.cuda.is_available()}")
        print(f"mps available: {torch.backends.mps.is_available()}")
        print(f"torch version: {torch.__version__}")
        print(f"numpy version: {np.__version__}")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Device cuda not available, falling back to cpu.")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("Device mps not available, falling back to cpu.")
        args.device = "cpu"
    print("Using device:", args.device)

    if args.mode == "selfplay":
        hexz.record_examples_mp(args)
    elif args.mode == "train":
        hexz.train_model(args)
    elif args.mode == "generate":
        hexz.generate_model(args)
    elif args.mode == "export":
        hexz.export_model(args)
    elif args.mode == "print":
        hexz.print_model_info(args)
    elif args.mode == "hello":
        print("Hello from hexz!")


if __name__ == "__main__":
    main()
