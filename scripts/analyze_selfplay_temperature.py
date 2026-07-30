#!/usr/bin/env python3
"""Diagnose stochastic self-play move selection from archived games.

The analysis is read-only. It compares each selected move with the root MCTS
visit-count argmax and estimates counterfactual selection behavior for fixed
temperatures and tau=1 -> tau=0 cutoff schedules.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pyhexz import hexz_pb2


PHASES = (
    ("moves_00_05", 0, 6),
    ("moves_06_11", 6, 12),
    ("moves_12_23", 12, 24),
    ("moves_24_47", 24, 48),
    ("moves_48_63", 48, 64),
    ("moves_64_plus", 64, math.inf),
)
TEMPERATURES = (1.0, 0.5, 0.25, 0.0)
DEFAULT_CUTOFFS = (6, 10, 12, 20, 30, 40, 50, 64)
DEFAULT_LATE_TEMPERATURES = (0.0, 0.25, 0.5)
EPSILON = 1e-7


def parse_int_set(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, stop_text = part.split("-", maxsplit=1)
            start, stop = int(start_text), int(stop_text)
            if start > stop:
                raise argparse.ArgumentTypeError(
                    f"invalid descending range: {part}"
                )
            result.update(range(start, stop + 1))
        else:
            result.add(int(part))
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError(
            "checkpoints must be non-negative integers or ranges"
        )
    return result


def parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(dict.fromkeys(int(part) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from exc
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError("values must be non-negative")
    return result


def parse_temperature_tuple(value: str) -> tuple[float, ...]:
    try:
        result = tuple(dict.fromkeys(float(part) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of temperatures"
        ) from exc
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError("temperatures must be non-negative")
    return result


def phase_name(move: int) -> str:
    for name, start, stop in PHASES:
        if start <= move < stop:
            return name
    raise AssertionError(f"no phase for move {move}")


def outcome_name(result: float) -> str:
    if result > 0:
        return "win"
    if result < 0:
        return "loss"
    return "draw"


def action_distribution(
    visit_distribution: np.ndarray, temperature: float
) -> np.ndarray:
    if temperature == 0:
        max_share = visit_distribution.max()
        is_max = np.isclose(
            visit_distribution, max_share, rtol=0, atol=EPSILON
        )
        return is_max.astype(np.float64) / is_max.sum()
    weights = np.power(
        visit_distribution.astype(np.float64), 1.0 / temperature
    )
    return weights / weights.sum()


def expected_selection_metrics(
    visit_distribution: np.ndarray, temperature: float
) -> tuple[float, float, float]:
    action_probs = action_distribution(visit_distribution, temperature)
    max_share = float(visit_distribution.max())
    below_max = visit_distribution < max_share - EPSILON
    probability_below_max = float(action_probs[below_max].sum())
    expected_selected_share = float(
        np.dot(action_probs, visit_distribution)
    )
    expected_regret = max_share - expected_selected_share
    return probability_below_max, expected_selected_share, expected_regret


@dataclass
class Aggregate:
    examples: int = 0
    below_argmax: int = 0
    below_half_argmax: int = 0
    below_quarter_argmax: int = 0
    selected_share_sum: float = 0
    max_share_sum: float = 0
    selected_q_sum: float = 0
    regrets: list[float] = field(default_factory=list)

    def add(
        self,
        *,
        selected_share: float,
        max_share: float,
        selected_q: float,
    ) -> None:
        self.examples += 1
        self.selected_share_sum += selected_share
        self.max_share_sum += max_share
        self.selected_q_sum += selected_q
        regret = max_share - selected_share
        self.regrets.append(regret)
        self.below_argmax += int(regret > EPSILON)
        self.below_half_argmax += int(
            selected_share < 0.5 * max_share - EPSILON
        )
        self.below_quarter_argmax += int(
            selected_share < 0.25 * max_share - EPSILON
        )

    def summary(self) -> dict:
        if self.examples == 0:
            return {"examples": 0}
        regrets = np.asarray(self.regrets)
        return {
            "examples": self.examples,
            "selected_below_argmax_rate": self.below_argmax / self.examples,
            "selected_below_half_argmax_rate": (
                self.below_half_argmax / self.examples
            ),
            "selected_below_quarter_argmax_rate": (
                self.below_quarter_argmax / self.examples
            ),
            "mean_selected_visit_share": (
                self.selected_share_sum / self.examples
            ),
            "mean_argmax_visit_share": self.max_share_sum / self.examples,
            "mean_selected_child_q": self.selected_q_sum / self.examples,
            "visit_share_regret": {
                "mean": float(regrets.mean()),
                "p50": float(np.quantile(regrets, 0.50)),
                "p90": float(np.quantile(regrets, 0.90)),
                "p95": float(np.quantile(regrets, 0.95)),
                "p99": float(np.quantile(regrets, 0.99)),
                "max": float(regrets.max()),
            },
        }


@dataclass
class ExpectedAggregate:
    examples: int = 0
    below_argmax_sum: float = 0
    selected_share_sum: float = 0
    regret_sum: float = 0

    def add(self, metrics: tuple[float, float, float]) -> None:
        below_argmax, selected_share, regret = metrics
        self.examples += 1
        self.below_argmax_sum += below_argmax
        self.selected_share_sum += selected_share
        self.regret_sum += regret

    def summary(self) -> dict:
        if self.examples == 0:
            return {"examples": 0}
        return {
            "examples": self.examples,
            "expected_below_argmax_rate": (
                self.below_argmax_sum / self.examples
            ),
            "expected_selected_visit_share": (
                self.selected_share_sum / self.examples
            ),
            "expected_visit_share_regret": self.regret_sum / self.examples,
        }


def decode_visit_distribution(example) -> np.ndarray:
    if example.encoding == hexz_pb2.TrainingExample.PYTORCH:
        tensor = torch.load(
            io.BytesIO(example.move_probs), weights_only=True
        )
        result = tensor.numpy().reshape(-1)
    elif example.encoding == hexz_pb2.TrainingExample.NUMPY:
        result = np.load(io.BytesIO(example.move_probs)).reshape(-1)
    else:
        raise ValueError(f"unsupported example encoding: {example.encoding}")
    total = float(result.sum())
    if not np.isfinite(result).all() or not np.isclose(total, 1, atol=1e-4):
        raise ValueError(f"invalid visit distribution with sum {total}")
    return result


def selected_tensor_index(example) -> int:
    move = example.move
    # The policy planes follow cpp::Move::Typ: flag=0, normal=1.
    plane = 0 if move.cell_type == hexz_pb2.Field.FLAG else 1
    if not (0 <= move.row < 11 and 0 <= move.col < 10):
        raise ValueError(f"invalid selected cell: ({move.row}, {move.col})")
    return plane * 11 * 10 + move.row * 10 + move.col


def checkpoint_from_relpath(relpath: str) -> int | None:
    parts = Path(relpath).parts
    if len(parts) < 4 or parts[0] != "checkpoints":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoints", required=True, type=parse_int_set)
    parser.add_argument(
        "--cutoffs",
        type=parse_int_tuple,
        default=DEFAULT_CUTOFFS,
        help="move cutoffs for tau=1 followed by each late temperature",
    )
    parser.add_argument(
        "--late-temperatures",
        type=parse_temperature_tuple,
        default=DEFAULT_LATE_TEMPERATURES,
        help="temperatures to simulate after each move cutoff",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="optional archive limit for smoke tests",
    )
    args = parser.parse_args()
    if args.max_games is not None and args.max_games <= 0:
        parser.error("--max-games must be positive")

    model_base = (
        Path(args.repo) / "models" / "flagz" / args.model
    )
    index_path = model_base / "index" / "requests.jsonl"
    if not index_path.is_file():
        parser.error(f"request index does not exist: {index_path}")

    index_entries = []
    with index_path.open(encoding="utf-8") as source:
        for lineno, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IOError(f"invalid index line {lineno}") from exc
            storage_checkpoint = checkpoint_from_relpath(entry["relpath"])
            if storage_checkpoint in args.checkpoints:
                index_entries.append(entry)
                if (
                    args.max_games is not None
                    and len(index_entries) >= args.max_games
                ):
                    break

    aggregates: dict[str, Aggregate] = {
        "overall": Aggregate(),
    }
    phase_aggregates: defaultdict[str, Aggregate] = defaultdict(Aggregate)
    outcome_aggregates: defaultdict[str, Aggregate] = defaultdict(Aggregate)
    phase_outcome_aggregates: defaultdict[
        tuple[str, str], Aggregate
    ] = defaultdict(Aggregate)
    checkpoint_aggregates: defaultdict[str, Aggregate] = defaultdict(Aggregate)
    all_temperatures = tuple(
        dict.fromkeys((*TEMPERATURES, *args.late_temperatures))
    )
    fixed_temperature = {
        temperature: ExpectedAggregate() for temperature in all_temperatures
    }
    schedules = {
        (cutoff, late_temperature): ExpectedAggregate()
        for cutoff in args.cutoffs
        for late_temperature in args.late_temperatures
    }
    examples_by_checkpoint: defaultdict[int, int] = defaultdict(int)
    skipped_examples = 0
    selected_share_mismatches = 0
    parsed_requests = 0

    for entry in index_entries:
        archive_path = model_base / entry["relpath"]
        with gzip.open(archive_path, "rb") as source:
            request = hexz_pb2.AddTrainingExamplesRequest()
            request.ParseFromString(source.read())
        parsed_requests += 1
        for example in request.examples:
            checkpoint = example.model_key.checkpoint
            if checkpoint not in args.checkpoints:
                skipped_examples += 1
                continue
            visit_distribution = decode_visit_distribution(example)
            selected_share = float(
                visit_distribution[selected_tensor_index(example)]
            )
            max_share = float(visit_distribution.max())
            child_visits = max(example.stats.visit_count - 1, 1)
            recorded_selected_share = (
                example.stats.selected_child_vc / child_visits
            )
            if not math.isclose(
                selected_share,
                recorded_selected_share,
                rel_tol=0,
                abs_tol=2e-3,
            ):
                selected_share_mismatches += 1

            values = {
                "selected_share": selected_share,
                "max_share": max_share,
                "selected_q": example.stats.selected_child_q,
            }
            aggregates["overall"].add(**values)
            phase = phase_name(example.move.move)
            outcome = outcome_name(example.result)
            phase_aggregates[phase].add(**values)
            outcome_aggregates[outcome].add(**values)
            phase_outcome_aggregates[(phase, outcome)].add(**values)
            checkpoint_aggregates[str(checkpoint)].add(**values)
            examples_by_checkpoint[checkpoint] += 1

            expected_by_temperature = {
                temperature: expected_selection_metrics(
                    visit_distribution, temperature
                )
                for temperature in all_temperatures
            }
            for temperature, aggregate in fixed_temperature.items():
                aggregate.add(expected_by_temperature[temperature])
            for (cutoff, late_temperature), aggregate in schedules.items():
                temperature = (
                    1.0
                    if example.move.move < cutoff
                    else late_temperature
                )
                aggregate.add(expected_by_temperature[temperature])

    result = {
        "scope": {
            "repo": str(Path(args.repo)),
            "model": args.model,
            "checkpoints": sorted(args.checkpoints),
            "indexed_archives": len(index_entries),
            "parsed_archives": parsed_requests,
            "examples": aggregates["overall"].examples,
            "examples_by_checkpoint": dict(
                sorted(examples_by_checkpoint.items())
            ),
            "skipped_examples_from_boundary_archives": skipped_examples,
            "selected_share_validation_mismatches": (
                selected_share_mismatches
            ),
        },
        "observed": {
            "overall": aggregates["overall"].summary(),
            "by_phase": {
                name: phase_aggregates[name].summary()
                for name, _, _ in PHASES
            },
            "by_outcome_from_player_to_move": {
                outcome: outcome_aggregates[outcome].summary()
                for outcome in ("win", "loss", "draw")
            },
            "by_phase_and_outcome_from_player_to_move": {
                phase: {
                    outcome: phase_outcome_aggregates[
                        (phase, outcome)
                    ].summary()
                    for outcome in ("win", "loss", "draw")
                }
                for phase, _, _ in PHASES
            },
            "by_checkpoint": {
                checkpoint: aggregate.summary()
                for checkpoint, aggregate in sorted(
                    checkpoint_aggregates.items(), key=lambda item: int(item[0])
                )
            },
        },
        "counterfactual": {
            "fixed_temperature_all_moves": {
                str(temperature): aggregate.summary()
                for temperature, aggregate in fixed_temperature.items()
            },
            "tau_1_then_late_temperature": {
                f"cutoff_{cutoff}_late_tau_{late_temperature}": (
                    aggregate.summary()
                )
                for (cutoff, late_temperature), aggregate in schedules.items()
            },
        },
        "notes": [
            "Rates are per position; positions within a game are correlated.",
            "Below-argmax means the sampled move had fewer root visits than the most-visited move; ties count as argmax.",
            "Visit-share regret is argmax visit share minus selected move visit share.",
            "Outcome correlation cannot by itself establish that a non-argmax move caused the result.",
            "Counterfactuals reuse recorded MCTS visit distributions and do not model how a different move changes later positions.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
