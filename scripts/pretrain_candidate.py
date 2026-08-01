#!/usr/bin/env python3
"""Run bounded offline training rounds for an initialized candidate model."""

import argparse
import json
import logging
from pathlib import Path

from pyhexz.config import TrainingConfig
from pyhexz.modelrepo import LocalModelRepository
from pyhexz.training import TrainingTask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batches", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--replay-window", type=int, default=1_025_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.rounds <= 0 or args.batches <= 0 or args.batch_size <= 0:
        parser.error("rounds, batches, and batch-size must be positive")
    required = args.batches * args.batch_size
    if required > args.replay_window:
        parser.error("one round cannot sample more than the replay window")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("pretrain_candidate")
    repo = LocalModelRepository(args.repo)
    try:
        available = repo.h5_size(args.model)
        if available < args.replay_window:
            raise ValueError(
                f"Model has {available} replay rows, fewer than {args.replay_window}"
            )
        starting_checkpoint = repo.get_latest_checkpoint(args.model)
        if starting_checkpoint is None:
            raise ValueError(f"Model {args.model!r} has no initial checkpoint")

        config = TrainingConfig(
            model_repo_base_dir=args.repo,
            model_name=args.model,
            model_type="resnet",
            model_blocks=10,
            batch_size=args.batch_size,
            training_trigger_threshold=25_000,
            training_examples_window_size=args.replay_window,
            num_epochs=1,
            training_batches_per_trigger=args.batches,
            replay_sampling_chunk_size=256,
            training_seed=args.training_seed,
            optimizer="adam",
            learning_rate=args.learning_rate,
            adam_weight_decay=args.weight_decay,
            device=args.device,
            shuffle=True,
            pin_memory=False,
        )

        completed = []
        checkpoint = starting_checkpoint
        for round_number in range(1, args.rounds + 1):
            result = TrainingTask(
                model_name=args.model,
                checkpoint=checkpoint,
                model_repo=repo,
                config=config,
                logger=logger,
            ).execute()
            checkpoint = result.model.checkpoint
            completed.append(
                {
                    "round": round_number,
                    "checkpoint": checkpoint,
                    "examples_trained": result.examples_trained,
                    "training_time_seconds": result.training_time,
                    "replay_passes": result.examples_trained / args.replay_window,
                }
            )
            print(json.dumps(completed[-1]), flush=True)

        summary = {
            "model": args.model,
            "starting_checkpoint": starting_checkpoint,
            "final_checkpoint": checkpoint,
            "rounds": completed,
        }
        output = (
            Path(args.repo)
            / "models"
            / "flagz"
            / args.model
            / "pretraining.json"
        )
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    finally:
        repo.close_all()


if __name__ == "__main__":
    main()
