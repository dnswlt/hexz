#!/usr/bin/env python3
"""Train an isolated checkpoint clone on frozen replay plus deep-search targets."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import uuid

import h5py
import numpy as np

from eval_checkpoints import evaluate
from frozen_replay_retrain import (
    H5_DATASETS,
    copy_replay_range,
    sha256,
    source_snapshot,
)
from pyhexz.config import TrainingConfig
from pyhexz.modelrepo import LocalModelRepository
from pyhexz.training import TrainingTask


def append_teacher_rows(
    h5_path: Path,
    teacher: dict[str, np.ndarray],
    train_indices: np.ndarray,
    repeat: int,
    source_checkpoint: int,
) -> int:
    appended = len(train_indices) * repeat
    with h5py.File(h5_path, "r+") as h5:
        initial = len(h5["boards"])
        for name in H5_DATASETS:
            h5[name].resize((initial + appended, *h5[name].shape[1:]))
        for repetition in range(repeat):
            start = initial + repetition * len(train_indices)
            stop = start + len(train_indices)
            h5["boards"][start:stop] = teacher["boards"][train_indices]
            h5["action_masks"][start:stop] = teacher["action_masks"][train_indices]
            h5["move_probs"][start:stop] = teacher["move_probs"][train_indices]
            h5["values"][start:stop] = teacher["values"][train_indices]
            h5["checkpoints"][start:stop] = source_checkpoint
        h5.flush()
    return appended


def load_teacher(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        required = {
            "boards",
            "action_masks",
            "move_probs",
            "values",
            "moves",
            "position_ids",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"teacher archive is missing arrays: {sorted(missing)}")
        teacher = {name: archive[name] for name in required}
    count = len(teacher["boards"])
    if count < 10 or any(len(value) != count for value in teacher.values()):
        raise ValueError("teacher arrays are empty or have inconsistent lengths")
    if teacher["boards"].shape[1:] != (11, 11, 10):
        raise ValueError(f"invalid teacher board shape: {teacher['boards'].shape}")
    if teacher["action_masks"].shape[1:] != (2, 11, 10):
        raise ValueError("invalid teacher action-mask shape")
    if teacher["move_probs"].shape[1:] != (2, 11, 10):
        raise ValueError("invalid teacher policy shape")
    if not np.allclose(teacher["move_probs"].sum(axis=(1, 2, 3)), 1, atol=1e-5):
        raise ValueError("teacher policies are not normalized")
    if np.min(teacher["moves"]) < 12:
        raise ValueError("teacher archive contains opening targets before move 12")
    return teacher


def replay_validation_arrays(
    source_h5: Path,
    start: int,
    end: int,
    count: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(np.arange(start, end), size=count, replace=False))
    with h5py.File(source_h5, "r") as h5:
        return tuple(
            h5[name][indices]
            for name in ("boards", "action_masks", "move_probs", "values")
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/dw/data/hexz-models"))
    parser.add_argument("--source-model", default="res10-r4-cp62")
    parser.add_argument("--source-checkpoint", type=int, default=50)
    parser.add_argument("--candidate-model", default="res10-r4-rean-cp50")
    parser.add_argument("--teacher-examples", type=Path, required=True)
    parser.add_argument("--teacher-holdout-fraction", type=float, default=0.2)
    parser.add_argument("--teacher-repeat", type=int, default=128)
    parser.add_argument("--batches", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--replay-window", type=int, default=2**20)
    parser.add_argument("--replay-chunk-size", type=int, default=256)
    parser.add_argument("--training-seed", type=int, default=51)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--validation-examples", type=int, default=4096)
    parser.add_argument("--validation-seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.source_checkpoint < 0 or args.batches <= 0 or args.batch_size <= 0:
        parser.error("checkpoint must be non-negative; batches and batch size must be positive")
    if not 0 < args.teacher_holdout_fraction < 0.5:
        parser.error("--teacher-holdout-fraction must be between 0 and 0.5")
    if args.teacher_repeat <= 0 or args.validation_examples <= 0:
        parser.error("teacher repeat and validation examples must be positive")
    if args.source_model == args.candidate_model:
        parser.error("source and candidate model names must differ")
    return args


def main() -> None:
    args = parse_args()
    models_base = args.repo / "models" / "flagz"
    source_base = models_base / args.source_model
    candidate_base = models_base / args.candidate_model
    source_h5 = source_base / "h5" / "examples.h5"
    if candidate_base.exists():
        raise FileExistsError(f"candidate already exists: {candidate_base}")

    teacher = load_teacher(args.teacher_examples)
    rng = np.random.default_rng(args.validation_seed)
    permutation = rng.permutation(len(teacher["boards"]))
    holdout_count = max(1, round(len(permutation) * args.teacher_holdout_fraction))
    holdout_indices = np.sort(permutation[:holdout_count])
    train_indices = np.sort(permutation[holdout_count:])

    source_before = source_snapshot(source_h5)
    repo = LocalModelRepository(str(args.repo))
    source_model = repo.get_model(
        args.source_model, args.source_checkpoint, map_location="cpu"
    )
    source_state = repo.get_training_state(
        args.source_model, args.source_checkpoint, map_location="cpu"
    )
    if source_state is None:
        raise ValueError("source checkpoint has no optimizer state")
    source_window = source_state.get("last_training_run", {}).get("examples_window")
    if not source_window or source_window[1] - source_window[0] != args.replay_window:
        raise ValueError(
            f"source checkpoint does not record the expected replay window: {source_window}"
        )
    replay_start, replay_end = source_window
    if args.validation_examples > args.replay_window:
        raise ValueError("validation sample exceeds replay window")

    replay_validation = replay_validation_arrays(
        source_h5,
        replay_start,
        replay_end,
        args.validation_examples,
        args.validation_seed,
    )
    teacher_holdout = tuple(
        teacher[name][holdout_indices]
        for name in ("boards", "action_masks", "move_probs", "values")
    )
    baseline_metrics = {
        "frozen_replay": evaluate(
            source_model, replay_validation, min(1024, args.validation_examples), args.device
        ),
        "teacher_holdout": evaluate(
            source_model, teacher_holdout, min(256, holdout_count), args.device
        ),
    }
    source_model = source_model.to("cpu")

    staging_name = f".{args.candidate_model}.reanalysis-{uuid.uuid4().hex}"
    staging_base = models_base / staging_name
    manifest = {
        "kind": "reanalysis_student",
        "status": "creating",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "model": args.source_model,
            "checkpoint": args.source_checkpoint,
            "replay_range": [replay_start, replay_end],
            "replay_snapshot": source_before,
        },
        "teacher": {
            "path": str(args.teacher_examples),
            "sha256": sha256(args.teacher_examples),
            "examples": len(teacher["boards"]),
            "train_examples": len(train_indices),
            "holdout_examples": len(holdout_indices),
            "repeat": args.teacher_repeat,
        },
        "candidate": {"model": args.candidate_model, "initial_checkpoint": 0},
        "training": {
            "batches": args.batches,
            "batch_size": args.batch_size,
            "training_seed": args.training_seed,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "replay_window": args.replay_window,
        },
        "baseline_metrics": baseline_metrics,
    }

    try:
        repo.store_model(staging_name, 0, source_model, training_state=source_state)
        repo.close_all()
        copy_replay_range(
            source_h5,
            staging_base / "h5" / "examples.h5",
            replay_start,
            replay_end,
        )
        appended = append_teacher_rows(
            staging_base / "h5" / "examples.h5",
            teacher,
            train_indices,
            args.teacher_repeat,
            args.source_checkpoint,
        )
        manifest["teacher"]["appended_rows"] = appended
        manifest["teacher"]["effective_fraction"] = appended / args.replay_window
        manifest["source"]["effective_base_range"] = [
            replay_start + appended,
            replay_end,
        ]
        (staging_base / "experiment.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(staging_base, candidate_base)
    except Exception:
        repo.close_all()
        shutil.rmtree(staging_base, ignore_errors=True)
        raise

    if source_snapshot(source_h5) != source_before:
        raise RuntimeError("source replay changed while creating candidate")

    config = TrainingConfig(
        model_repo_base_dir=str(args.repo),
        model_name=args.candidate_model,
        model_type="resnet",
        model_blocks=10,
        batch_size=args.batch_size,
        training_trigger_threshold=25_000,
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
    candidate_repo = LocalModelRepository(str(args.repo))
    try:
        result = TrainingTask(
            model_name=args.candidate_model,
            checkpoint=0,
            model_repo=candidate_repo,
            config=config,
            logger=logging.getLogger("reanalysis_student"),
        ).execute()
        trained_model = candidate_repo.get_model(
            args.candidate_model, result.model.checkpoint, map_location="cpu"
        )
        student_metrics = {
            "frozen_replay": evaluate(
                trained_model,
                replay_validation,
                min(1024, args.validation_examples),
                args.device,
            ),
            "teacher_holdout": evaluate(
                trained_model,
                teacher_holdout,
                min(256, holdout_count),
                args.device,
            ),
        }
    finally:
        candidate_repo.close_all()

    if source_snapshot(source_h5) != source_before:
        raise RuntimeError("source replay changed during isolated training")
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "candidate": {
                **manifest["candidate"],
                "trained_checkpoint": result.model.checkpoint,
            },
            "result": dataclasses.asdict(result),
            "student_metrics": student_metrics,
            "verification": {"source_replay_unchanged": True},
        }
    )
    (candidate_base / "experiment.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
