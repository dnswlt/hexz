#!/usr/bin/env python3
"""Evaluate checkpoints on one deterministic replay sample."""

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from pyhexz.modelrepo import LocalModelRepository


def parse_checkpoints(value: str) -> list[int]:
    checkpoints = [int(part) for part in value.split(",")]
    if not checkpoints or any(checkpoint < 0 for checkpoint in checkpoints):
        raise argparse.ArgumentTypeError("checkpoints must be non-negative integers")
    return list(dict.fromkeys(checkpoints))


def evaluate(model, arrays, batch_size: int, device: str) -> dict:
    boards, masks, move_probs, values = arrays
    model = model.to(device)
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    mse = nn.MSELoss(reduction="sum")
    policy_sum = 0.0
    value_sum = 0.0
    sign_correct = 0
    bias_sum = 0.0
    count = len(values)

    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            xb = torch.from_numpy(boards[start:stop]).to(device)
            xm = torch.from_numpy(masks[start:stop]).to(device)
            yp = torch.from_numpy(move_probs[start:stop]).to(device)
            yv = torch.from_numpy(values[start:stop]).to(device)
            pp, pv = model(xb, xm)
            policy_sum += ce(pp, yp.flatten(1)).item()
            value_sum += mse(pv, yv).item()
            sign_correct += (torch.sign(pv) == torch.sign(yv)).sum().item()
            bias_sum += (pv - yv).sum().item()

    # Use the first evaluation batch for a consistent gradient health check.
    model.zero_grad(set_to_none=True)
    xb = torch.from_numpy(boards[:batch_size]).to(device)
    xm = torch.from_numpy(masks[:batch_size]).to(device)
    yp = torch.from_numpy(move_probs[:batch_size]).to(device)
    yv = torch.from_numpy(values[:batch_size]).to(device)
    pp, pv = model(xb, xm)
    loss = nn.CrossEntropyLoss()(pp, yp.flatten(1)) + nn.MSELoss()(pv, yv)
    loss.backward()

    parameter_sq = 0.0
    gradient_sq = 0.0
    parameter_tensors = 0
    gradient_tensors = 0
    nonzero_gradient_tensors = 0
    finite_gradients = True
    for parameter in model.parameters():
        parameter_tensors += 1
        parameter_sq += parameter.detach().float().square().sum().item()
        if parameter.grad is None:
            continue
        gradient_tensors += 1
        gradient = parameter.grad.detach().float()
        finite_gradients = finite_gradients and torch.isfinite(gradient).all().item()
        gradient_sq += gradient.square().sum().item()
        nonzero_gradient_tensors += int(torch.count_nonzero(gradient).item() > 0)

    gradient_norm = math.sqrt(gradient_sq)
    parameter_norm = math.sqrt(parameter_sq)
    return {
        "policy_cross_entropy": policy_sum / count,
        "value_mse": value_sum / count,
        "value_sign_accuracy": sign_correct / count,
        "value_bias": bias_sum / count,
        "gradient_loss": loss.item(),
        "gradient_norm": gradient_norm,
        "parameter_norm": parameter_norm,
        "gradient_parameter_ratio": gradient_norm / (parameter_norm + 1e-8),
        "parameter_tensors": parameter_tensors,
        "gradient_tensors": gradient_tensors,
        "nonzero_gradient_tensors": nonzero_gradient_tensors,
        "finite_gradients": finite_gradients,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--replay-model",
        help="Read the evaluation replay from this model (default: --model)",
    )
    parser.add_argument("--checkpoints", required=True, type=parse_checkpoints)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample-population", type=int, default=1_025_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.examples <= 0 or args.examples > args.sample_population:
        parser.error("--examples must be between 1 and --sample-population")
    if args.batch_size <= 0 or args.batch_size > args.examples:
        parser.error("--batch-size must be between 1 and --examples")

    rng = np.random.default_rng(args.seed)
    indices = np.sort(
        rng.choice(args.sample_population, size=args.examples, replace=False)
    )
    replay_model = args.replay_model or args.model
    h5_path = (
        Path(args.repo)
        / "models"
        / "flagz"
        / replay_model
        / "h5"
        / "examples.h5"
    )
    with h5py.File(h5_path, "r") as h5:
        if args.sample_population > len(h5["boards"]):
            parser.error("--sample-population exceeds the replay size")
        arrays = tuple(
            h5[name][indices]
            for name in ("boards", "action_masks", "move_probs", "values")
        )

    repo = LocalModelRepository(args.repo)
    result = {
        "sample": {
            "replay_model": replay_model,
            "seed": args.seed,
            "population": args.sample_population,
            "examples": args.examples,
            "first_index": int(indices[0]),
            "last_index": int(indices[-1]),
        }
    }
    for checkpoint in args.checkpoints:
        model = repo.get_model(args.model, checkpoint, map_location="cpu")
        result[f"checkpoint_{checkpoint}"] = evaluate(
            model, arrays, args.batch_size, args.device
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
