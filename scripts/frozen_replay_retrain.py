#!/usr/bin/env python3
"""Train an isolated candidate on an exact, frozen replay range."""

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import uuid

import h5py
import torch

from pyhexz.config import TrainingConfig
from pyhexz.modelrepo import LocalModelRepository
from pyhexz.training import TrainingTask


H5_DATASETS = ("boards", "action_masks", "move_probs", "values", "checkpoints")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot(path: Path) -> dict:
    stat = path.stat()
    with h5py.File(path, "r") as h5:
        lengths = {name: len(h5[name]) for name in H5_DATASETS}
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
        "dataset_lengths": lengths,
    }


def copy_replay_range(
    source_path: Path,
    destination_path: Path,
    start: int,
    end: int,
    copy_chunk_size: int = 4096,
) -> None:
    count = end - start
    if count <= 0:
        raise ValueError("Replay range must be non-empty")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source:
        missing = [name for name in H5_DATASETS if name not in source]
        if missing:
            raise ValueError(f"Source replay is missing datasets: {missing}")
        lengths = {name: len(source[name]) for name in H5_DATASETS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Source replay dataset lengths differ: {lengths}")
        total = next(iter(lengths.values()))
        if start < 0 or end > total:
            raise ValueError(
                f"Replay range [{start}, {end}) is outside source size {total}"
            )

        with h5py.File(destination_path, "w") as destination:
            for name in H5_DATASETS:
                src = source[name]
                kwargs = {
                    "shape": (count, *src.shape[1:]),
                    "maxshape": (None, *src.shape[1:]),
                    "dtype": src.dtype,
                    "chunks": src.chunks,
                    "compression": src.compression,
                }
                if src.compression is not None:
                    kwargs["compression_opts"] = src.compression_opts
                dst = destination.create_dataset(name, **kwargs)
                for offset in range(0, count, copy_chunk_size):
                    size = min(copy_chunk_size, count - offset)
                    dst[offset : offset + size] = src[
                        start + offset : start + offset + size
                    ]
            destination.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=int)
    parser.add_argument("--comparison-checkpoint", required=True, type=int)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--replay-start", required=True, type=int)
    parser.add_argument("--replay-end", required=True, type=int)
    parser.add_argument("--batches", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--replay-window", type=int, default=2**20)
    parser.add_argument("--replay-chunk-size", type=int, default=256)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--trigger-threshold", type=int, default=25_000)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    if args.source_model == args.candidate_model:
        raise ValueError("Source and candidate model names must differ")
    if args.replay_end - args.replay_start != args.replay_window:
        raise ValueError(
            "The exact replay range length must equal --replay-window"
        )

    repo_base = Path(args.repo)
    models_base = repo_base / "models" / "flagz"
    source_base = models_base / args.source_model
    candidate_base = models_base / args.candidate_model
    source_h5 = source_base / "h5" / "examples.h5"
    if candidate_base.exists():
        raise FileExistsError(f"Candidate already exists: {candidate_base}")

    before = source_snapshot(source_h5)
    staging_name = f".{args.candidate_model}.frozen-{uuid.uuid4().hex}"
    staging_base = models_base / staging_name
    repo = LocalModelRepository(str(repo_base))

    try:
        source_model = repo.get_model(
            args.source_model, args.source_checkpoint, map_location="cpu"
        )
        source_state = repo.get_training_state(
            args.source_model, args.source_checkpoint, map_location="cpu"
        )
        if source_state is None:
            raise ValueError("Source checkpoint has no optimizer training state")
        repo.store_model(
            staging_name,
            0,
            source_model,
            training_state=source_state,
        )
        repo.close_all()

        copy_replay_range(
            source_h5,
            staging_base / "h5" / "examples.h5",
            args.replay_start,
            args.replay_end,
        )
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": "frozen_replay_retrain",
            "source": {
                "model": args.source_model,
                "checkpoint": args.source_checkpoint,
                "comparison_checkpoint": args.comparison_checkpoint,
                "replay_range": [args.replay_start, args.replay_end],
                "replay_snapshot": before,
            },
            "candidate": {"model": args.candidate_model, "checkpoint": 0},
            "training": {
                "batches": args.batches,
                "batch_size": args.batch_size,
                "replay_window": args.replay_window,
                "replay_chunk_size": args.replay_chunk_size,
                "training_seed": args.training_seed,
                "trigger_threshold": args.trigger_threshold,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "device": args.device,
            },
        }
        (staging_base / "experiment.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(staging_base, candidate_base)
    except Exception:
        repo.close_all()
        shutil.rmtree(staging_base, ignore_errors=True)
        raise

    after_copy = source_snapshot(source_h5)
    if after_copy != before:
        raise RuntimeError("Source replay changed while creating the candidate")

    config = TrainingConfig(
        model_repo_base_dir=str(repo_base),
        model_name=args.candidate_model,
        model_type="resnet",
        model_blocks=10,
        batch_size=args.batch_size,
        training_trigger_threshold=args.trigger_threshold,
        training_examples_window_size=args.replay_window,
        num_epochs=1,
        training_batches_per_trigger=args.batches,
        replay_sampling_chunk_size=args.replay_chunk_size,
        training_seed=args.training_seed,
        optimizer="adam",
        learning_rate=args.learning_rate,
        adam_weight_decay=args.weight_decay,
        device=args.device,
        shuffle=True,
        pin_memory=False,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("frozen_replay_retrain")
    candidate_repo = LocalModelRepository(str(repo_base))
    result = TrainingTask(
        model_name=args.candidate_model,
        checkpoint=0,
        model_repo=candidate_repo,
        config=config,
        logger=logger,
    ).execute()
    candidate_repo.close_all()

    comparison_state = repo.get_training_state(
        args.source_model, args.comparison_checkpoint, map_location="cpu"
    )
    trained_state = repo.get_training_state(
        args.candidate_model, result.model.checkpoint, map_location="cpu"
    )
    if comparison_state is None or trained_state is None:
        raise RuntimeError("Missing comparison or trained optimizer state")
    comparison_ranges = comparison_state["last_training_run"]["sampled_ranges"]
    expected_prefix = [
        (start - args.replay_start, stop - args.replay_start)
        for start, stop in comparison_ranges
    ]
    actual_ranges = trained_state["last_training_run"]["sampled_ranges"]
    prefix_matches = actual_ranges[: len(expected_prefix)] == expected_prefix

    after_training = source_snapshot(source_h5)
    if after_training != before:
        raise RuntimeError("Source replay changed during isolated training")

    manifest["candidate"]["trained_checkpoint"] = result.model.checkpoint
    manifest["result"] = dataclasses.asdict(result)
    manifest["verification"] = {
        "source_replay_unchanged": True,
        "comparison_sampled_ranges": len(expected_prefix),
        "trained_sampled_ranges": len(actual_ranges),
        "comparison_prefix_matches": prefix_matches,
    }
    (candidate_base / "experiment.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
