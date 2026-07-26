"""Utilities for creating isolated, reproducible training experiments."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import h5py

from pyhexz.modelrepo import LocalModelRepository


_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_H5_DATASETS = ("boards", "action_masks", "move_probs", "values", "checkpoints")


def _validate_model_name(name: str) -> None:
    if not _MODEL_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid model name {name!r}; use letters, digits, '.', '_', or '-'"
        )


def _copy_replay_tail(
    source: h5py.File,
    destination: h5py.File,
    count: int,
    copy_chunk_size: int = 4096,
) -> tuple[int, int]:
    missing = [name for name in _H5_DATASETS if name not in source]
    if missing:
        raise ValueError(f"Source replay is missing datasets: {missing}")
    lengths = {name: len(source[name]) for name in _H5_DATASETS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Source replay dataset lengths differ: {lengths}")
    total = next(iter(lengths.values()))
    if count <= 0 or count > total:
        raise ValueError(f"Cannot copy {count} examples from replay of size {total}")

    start = total - count
    for name in _H5_DATASETS:
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
            dst[offset : offset + size] = src[start + offset : start + offset + size]
    destination.flush()
    return start, total


def initialize_training_experiment(
    repo_base_dir: str | os.PathLike,
    source_model: str,
    source_checkpoint: int,
    candidate_model: str,
    replay_examples: int,
    trigger_threshold: int,
) -> dict:
    """Seeds a candidate model without modifying its source.

    The candidate starts at checkpoint 0 with weights copied from
    source_model:source_checkpoint. Its replay is the newest requested number of
    source examples, rounded down to a multiple of trigger_threshold. Aligning
    the replay ensures the training server waits for one complete fresh-data
    interval before creating the first candidate checkpoint.
    """

    _validate_model_name(source_model)
    _validate_model_name(candidate_model)
    if source_model == candidate_model:
        raise ValueError("Source and candidate model names must differ")
    if source_checkpoint < 0:
        raise ValueError("Source checkpoint must be non-negative")
    if replay_examples <= 0:
        raise ValueError("Replay example count must be positive")
    if trigger_threshold <= 0:
        raise ValueError("Trigger threshold must be positive")

    aligned_examples = replay_examples - replay_examples % trigger_threshold
    if aligned_examples == 0:
        raise ValueError(
            "Replay example count must contain at least one complete trigger interval"
        )

    repo_base = Path(repo_base_dir)
    models_base = repo_base / "models" / "flagz"
    candidate_base = models_base / candidate_model
    if candidate_base.exists():
        raise FileExistsError(f"Candidate model already exists: {candidate_base}")

    models_base.mkdir(parents=True, exist_ok=True)
    staging_model = f".{candidate_model}.seed-{uuid.uuid4().hex}"
    staging_base = models_base / staging_model
    repo = LocalModelRepository(str(repo_base))

    try:
        source = repo.get_model(source_model, source_checkpoint, map_location="cpu")
        repo.store_model(staging_model, 0, source)

        with repo.acquire_h5(source_model) as source_h5:
            with repo.acquire_h5(staging_model) as destination_h5:
                replay_start, replay_end = _copy_replay_tail(
                    source_h5, destination_h5, aligned_examples
                )
        repo.close_all()

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "model": source_model,
                "checkpoint": source_checkpoint,
                "replay_range": [replay_start, replay_end],
            },
            "candidate": {
                "model": candidate_model,
                "checkpoint": 0,
            },
            "training": {
                "trigger_threshold": trigger_threshold,
                "requested_replay_examples": replay_examples,
                "seed_replay_examples": aligned_examples,
            },
        }
        with open(staging_base / "experiment.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        os.rename(staging_base, candidate_base)
        return manifest
    except Exception:
        repo.close_all()
        shutil.rmtree(staging_base, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an isolated Hexz training candidate from an existing checkpoint."
    )
    parser.add_argument("--repo", required=True, help="Model repository base directory")
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=int)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--replay-examples", type=int, default=2**20)
    parser.add_argument("--trigger-threshold", type=int, default=25_000)
    args = parser.parse_args()

    manifest = initialize_training_experiment(
        repo_base_dir=args.repo,
        source_model=args.source_model,
        source_checkpoint=args.source_checkpoint,
        candidate_model=args.candidate_model,
        replay_examples=args.replay_examples,
        trigger_threshold=args.trigger_threshold,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
