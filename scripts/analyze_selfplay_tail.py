#!/usr/bin/env python3
"""Analyze safe separated-tail truncation on archived Flagz self-play games.

This script is read-only.  For each game it finds the first position after all
flags are gone where conservative future-reachability sets no longer overlap.
It then tries to solve both independent single-player tails exactly and
compares the predicted optimal winner with the recorded self-play result.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pyhexz import hexz_pb2
from pyhexz.flagz_tail import (
    CELL_INDEX,
    NUM_CELLS,
    VALID_CELLS,
    ExactTailSolver,
    PlayerTailState,
    Reachability,
    TailSolveLimit,
    optimistic_reachability,
    play_normal,
)


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


def checkpoint_from_relpath(relpath: str) -> int | None:
    parts = Path(relpath).parts
    if len(parts) < 4 or parts[0] != "checkpoints":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def decode_board(example) -> np.ndarray:
    if example.encoding == hexz_pb2.TrainingExample.PYTORCH:
        tensor = torch.load(io.BytesIO(example.board), weights_only=True)
        result = tensor.numpy()
    elif example.encoding == hexz_pb2.TrainingExample.NUMPY:
        result = np.load(io.BytesIO(example.board))
    else:
        raise ValueError(f"unsupported example encoding: {example.encoding}")
    if result.shape != (11, 11, 10):
        raise ValueError(f"invalid board shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("board contains non-finite values")
    return result


def player_state(board: np.ndarray, player: int) -> PlayerTailState:
    offset = 5 * player
    blocked = 0
    next_values = bytearray(NUM_CELLS)
    grass = bytearray(NUM_CELLS)
    for cell, (r, c) in enumerate(VALID_CELLS):
        if board[2 + offset, r, c]:
            blocked |= 1 << cell
        next_values[cell] = round(float(board[3 + offset, r, c]))
        grass[cell] = round(float(board[10, r, c]))
    return PlayerTailState(blocked, bytes(next_values), bytes(grass))


def score(board: np.ndarray, player: int) -> int:
    return round(float(board[1 + 5 * player].sum()))


def flags(board: np.ndarray, player: int) -> int:
    return round(float(board[4 + 5 * player, 0, 0]))


def quantiles(values: list[float | int]) -> dict:
    if not values:
        return {"count": 0}
    data = np.asarray(values)
    return {
        "count": len(values),
        "min": float(data.min()),
        "p25": float(np.quantile(data, 0.25)),
        "p50": float(np.quantile(data, 0.50)),
        "p75": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(data.max()),
        "mean": float(data.mean()),
    }


def result_sign(value: float | int) -> int:
    return int(value > 0) - int(value < 0)


@dataclass
class Separation:
    index: int
    move: int
    board: np.ndarray
    states: tuple[PlayerTailState, PlayerTailState]
    reachability: tuple[Reachability, Reachability]


def first_separation(
    examples,
) -> tuple[Separation | None, int, int, int]:
    positions_after_flags = 0
    path_states = 0
    relaxed_fallbacks = 0
    for index, example in enumerate(examples):
        board = decode_board(example)
        if flags(board, 0) or flags(board, 1):
            continue
        positions_after_flags += 1
        states = (player_state(board, 0), player_state(board, 1))
        reachability = tuple(
            optimistic_reachability(state) for state in states
        )
        path_states += sum(item.path_states for item in reachability)
        relaxed_fallbacks += sum(
            item.used_relaxed_fallback for item in reachability
        )
        if not reachability[0].cells.intersection(reachability[1].cells):
            return (
                Separation(
                    index=index,
                    move=example.move.move,
                    board=board,
                    states=states,
                    reachability=reachability,
                ),
                positions_after_flags,
                path_states,
                relaxed_fallbacks,
            )
    return None, positions_after_flags, path_states, relaxed_fallbacks


def validate_next_transition(examples, separation: Separation) -> str:
    index = separation.index
    if index + 1 >= len(examples):
        return "last_position"
    example = examples[index]
    if example.move.cell_type != hexz_pb2.Field.NORMAL:
        return "not_normal"
    next_example = examples[index + 1]
    if next_example.move.move != example.move.move + 1:
        return "move_gap"

    cell = CELL_INDEX[(example.move.row, example.move.col)]
    expected, gained = play_normal(separation.states[0], cell)
    next_board = decode_board(next_example)
    mover_in_next_view = int(next_example.turn != example.turn)
    observed = player_state(next_board, mover_in_next_view)
    observed_gain = (
        score(next_board, mover_in_next_view) - score(separation.board, 0)
    )
    if expected != observed or gained != observed_gain:
        return "mismatch"
    return "matched"


def actual_final_scores(examples, cutoff_index: int) -> tuple[list[int], int]:
    """Returns final scores oriented like the cutoff and its recorded result."""

    cutoff = examples[cutoff_index]
    last = examples[-1]
    if last.move.cell_type != hexz_pb2.Field.NORMAL:
        raise ValueError("a post-separation game ended with a flag move")
    board = decode_board(last)
    state = player_state(board, 0)
    cell = CELL_INDEX[(last.move.row, last.move.col)]
    _, gained = play_normal(state, cell)
    scores_in_last_view = [score(board, 0) + gained, score(board, 1)]
    if last.turn == cutoff.turn:
        scores_at_cutoff = scores_in_last_view
    else:
        scores_at_cutoff = list(reversed(scores_in_last_view))
    return scores_at_cutoff, result_sign(cutoff.result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoints", required=True, type=parse_int_set)
    parser.add_argument(
        "--max-games",
        type=int,
        help="optional archive limit for smoke tests",
    )
    parser.add_argument(
        "--solver-node-limit",
        type=int,
        default=250_000,
        help="maximum memoized states per player",
    )
    parser.add_argument(
        "--solver-time-limit-ms",
        type=float,
        default=1_000,
        help="wall-clock limit per player",
    )
    args = parser.parse_args()
    if args.max_games is not None and args.max_games <= 0:
        parser.error("--max-games must be positive")
    if args.solver_node_limit <= 0:
        parser.error("--solver-node-limit must be positive")
    if args.solver_time_limit_ms <= 0:
        parser.error("--solver-time-limit-ms must be positive")

    model_base = Path(args.repo) / "models" / "flagz" / args.model
    index_path = model_base / "index" / "requests.jsonl"
    if not index_path.is_file():
        parser.error(f"request index does not exist: {index_path}")

    entries = []
    with index_path.open(encoding="utf-8") as source:
        for lineno, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IOError(f"invalid index line {lineno}") from exc
            if checkpoint_from_relpath(entry["relpath"]) in args.checkpoints:
                entries.append(entry)
                if args.max_games and len(entries) >= args.max_games:
                    break

    games = 0
    examples_total = 0
    search_micros_total = 0
    positions_after_flags = 0
    reachability_path_states = 0
    reachability_relaxed_fallbacks = 0
    separated_games = 0
    separated_at_move = []
    tail_examples = []
    tail_search_micros = []
    reachable_sizes = [[], []]
    separation_kinds = Counter()
    transition_validation = Counter()
    solved_games = 0
    solver_limits = Counter()
    winner_comparison = Counter()
    solver_states = []
    solver_seconds = []
    component_sizes = []
    score_margins_at_cutoff = []
    optimal_score_margins = []
    actual_score_margins = []
    optimal_minus_actual = [[], []]
    actual_result_validation = Counter()
    winner_transitions = Counter()
    margin_thresholds = (0, 1, 2, 3, 5, 10, 20)
    winner_by_optimal_margin = {
        threshold: Counter() for threshold in margin_thresholds
    }
    savings_by_optimal_margin = {
        threshold: Counter() for threshold in margin_thresholds
    }
    solved_tail_examples = 0
    solved_tail_search_micros = 0
    discrepancy_samples = []
    non_monotonic_samples = []
    started = time.monotonic()

    for entry in entries:
        request = hexz_pb2.AddTrainingExamplesRequest()
        archive_path = model_base / entry["relpath"]
        with gzip.open(archive_path, "rb") as source:
            request.ParseFromString(source.read())
        examples = request.examples
        if not examples:
            continue
        games += 1
        examples_total += len(examples)
        search_micros_total += sum(
            example.stats.duration_micros for example in examples
        )

        (
            separation,
            checked_after_flags,
            checked_path_states,
            checked_relaxed_fallbacks,
        ) = first_separation(examples)
        positions_after_flags += checked_after_flags
        reachability_path_states += checked_path_states
        reachability_relaxed_fallbacks += checked_relaxed_fallbacks
        if separation is None:
            continue
        separated_games += 1
        separated_at_move.append(separation.move)
        tail_examples.append(len(examples) - separation.index)
        tail_search_micros.append(
            sum(
                example.stats.duration_micros
                for example in examples[separation.index :]
            )
        )
        for player in range(2):
            reachable_sizes[player].append(
                len(separation.reachability[player].cells)
            )
        if not separation.reachability[1].cells:
            separation_kinds["opponent_has_no_reachable_cell"] += 1
        else:
            separation_kinds["both_players_have_reachable_cells"] += 1
        transition_validation[
            validate_next_transition(examples, separation)
        ] += 1

        for later_example in examples[separation.index + 1 :]:
            later_board = decode_board(later_example)
            later_states = (
                player_state(later_board, 0),
                player_state(later_board, 1),
            )
            later_reachability = tuple(
                optimistic_reachability(state) for state in later_states
            )
            if later_reachability[0].cells.intersection(
                later_reachability[1].cells
            ):
                if len(non_monotonic_samples) < 10:
                    non_monotonic_samples.append(
                        {
                            "game_id": request.game_id,
                            "separation_move": separation.move,
                            "later_move": later_example.move.move,
                        }
                    )
                break

        player_results = []
        solve_failed = False
        for state in separation.states:
            solver = ExactTailSolver(
                node_limit=args.solver_node_limit,
                time_limit_seconds=args.solver_time_limit_ms / 1_000,
            )
            try:
                addition = solver.solve(state)
            except TailSolveLimit as exc:
                solver_limits[str(exc)] += 1
                solve_failed = True
            finally:
                solver_states.append(solver.stats.states)
                solver_seconds.append(solver.stats.elapsed_seconds)
                component_sizes.extend(solver.stats.component_sizes)
            if solve_failed:
                break
            player_results.append(score(separation.board, player) + addition)

        if solve_failed:
            continue
        solved_games += 1
        solved_tail_examples += len(examples) - separation.index
        solved_tail_search_micros += sum(
            example.stats.duration_micros
            for example in examples[separation.index :]
        )
        predicted = result_sign(player_results[0] - player_results[1])
        recorded = result_sign(examples[separation.index].result)
        final_scores, final_recorded = actual_final_scores(
            examples, separation.index
        )
        reconstructed_result = result_sign(
            final_scores[0] - final_scores[1]
        )
        actual_result_validation[
            "matched" if reconstructed_result == final_recorded else "mismatch"
        ] += 1
        for player in range(2):
            optimal_minus_actual[player].append(
                player_results[player] - final_scores[player]
            )
        comparison = "match" if predicted == recorded else "mismatch"
        winner_comparison[comparison] += 1
        winner_transitions[f"{predicted}_to_{recorded}"] += 1
        absolute_optimal_margin = abs(player_results[0] - player_results[1])
        for threshold, counts in winner_by_optimal_margin.items():
            if absolute_optimal_margin >= threshold:
                counts[comparison] += 1
                savings = savings_by_optimal_margin[threshold]
                savings["games"] += 1
                savings["examples"] += len(examples) - separation.index
                savings["search_micros"] += sum(
                    example.stats.duration_micros
                    for example in examples[separation.index :]
                )
        score_margins_at_cutoff.append(
            score(separation.board, 0) - score(separation.board, 1)
        )
        optimal_score_margins.append(player_results[0] - player_results[1])
        actual_score_margins.append(final_scores[0] - final_scores[1])
        if comparison == "mismatch" and len(discrepancy_samples) < 20:
            discrepancy_samples.append(
                {
                    "game_id": request.game_id,
                    "archive": entry["relpath"],
                    "move": separation.move,
                    "checkpoint": examples[separation.index].model_key.checkpoint,
                    "scores_at_cutoff": [
                        score(separation.board, 0),
                        score(separation.board, 1),
                    ],
                    "optimal_final_scores": player_results,
                    "actual_final_scores": final_scores,
                    "optimal_minus_actual": [
                        player_results[player] - final_scores[player]
                        for player in range(2)
                    ],
                    "predicted_result": predicted,
                    "recorded_result": recorded,
                }
            )

    elapsed = time.monotonic() - started
    tail_examples_sum = sum(tail_examples)
    tail_micros_sum = sum(tail_search_micros)
    result = {
        "scope": {
            "repo": str(Path(args.repo)),
            "model": args.model,
            "checkpoints": sorted(args.checkpoints),
            "archives": len(entries),
            "games": games,
            "examples": examples_total,
            "search_duration_seconds": search_micros_total / 1_000_000,
            "analysis_duration_seconds": elapsed,
            "solver_node_limit_per_player": args.solver_node_limit,
            "solver_time_limit_ms_per_player": args.solver_time_limit_ms,
        },
        "separation": {
            "games": separated_games,
            "game_rate": separated_games / games if games else math.nan,
            "positions_checked_after_flags": positions_after_flags,
            "reachability_path_states": reachability_path_states,
            "reachability_relaxed_fallbacks": (
                reachability_relaxed_fallbacks
            ),
            "first_separated_move": quantiles(separated_at_move),
            "tail_examples_per_separated_game": quantiles(tail_examples),
            "reachable_cells_player_to_move": quantiles(reachable_sizes[0]),
            "reachable_cells_opponent": quantiles(reachable_sizes[1]),
            "kinds": dict(separation_kinds),
            "transition_validation": dict(transition_validation),
            "later_overlap_after_separation": len(non_monotonic_samples),
            "later_overlap_samples": non_monotonic_samples,
        },
        "potential_savings": {
            "examples": tail_examples_sum,
            "example_rate": (
                tail_examples_sum / examples_total
                if examples_total
                else math.nan
            ),
            "recorded_search_seconds": tail_micros_sum / 1_000_000,
            "recorded_search_time_rate": (
                tail_micros_sum / search_micros_total
                if search_micros_total
                else math.nan
            ),
            "with_python_solver_cap": {
                "examples": solved_tail_examples,
                "example_rate": (
                    solved_tail_examples / examples_total
                    if examples_total
                    else math.nan
                ),
                "recorded_search_seconds": (
                    solved_tail_search_micros / 1_000_000
                ),
                "recorded_search_time_rate": (
                    solved_tail_search_micros / search_micros_total
                    if search_micros_total
                    else math.nan
                ),
            },
        },
        "exact_solver": {
            "solved_games": solved_games,
            "solved_rate_of_separated": (
                solved_games / separated_games
                if separated_games
                else math.nan
            ),
            "limits": dict(solver_limits),
            "winner_comparison": dict(winner_comparison),
            "winner_transitions": dict(winner_transitions),
            "winner_match_rate": (
                winner_comparison["match"] / solved_games
                if solved_games
                else math.nan
            ),
            "states_per_attempted_player": quantiles(solver_states),
            "seconds_per_attempted_player": quantiles(solver_seconds),
            "component_sizes": quantiles(component_sizes),
            "score_margin_at_cutoff": quantiles(score_margins_at_cutoff),
            "optimal_score_margin": quantiles(optimal_score_margins),
            "actual_score_margin": quantiles(actual_score_margins),
            "optimal_minus_actual_score": {
                "player_to_move": quantiles(optimal_minus_actual[0]),
                "opponent": quantiles(optimal_minus_actual[1]),
            },
            "actual_result_reconstruction": dict(actual_result_validation),
            "winner_comparison_by_minimum_absolute_optimal_margin": {
                str(threshold): {
                    **dict(counts),
                    "match_rate": (
                        counts["match"] / sum(counts.values())
                        if counts
                        else math.nan
                    ),
                }
                for threshold, counts in winner_by_optimal_margin.items()
            },
            "savings_by_minimum_absolute_optimal_margin": {
                str(threshold): {
                    "games": savings["games"],
                    "examples": savings["examples"],
                    "example_rate": (
                        savings["examples"] / examples_total
                        if examples_total
                        else math.nan
                    ),
                    "recorded_search_seconds": (
                        savings["search_micros"] / 1_000_000
                    ),
                    "recorded_search_time_rate": (
                        savings["search_micros"] / search_micros_total
                        if search_micros_total
                        else math.nan
                    ),
                }
                for threshold, savings in savings_by_optimal_margin.items()
            },
            "mismatch_samples": discrepancy_samples,
        },
        "notes": [
            (
                "Board tensors are oriented to the player to move; player 0 "
                "in each reported cutoff is that player."
            ),
            (
                "Reachability is an over-approximation. It may miss safe "
                "cutoffs but cannot declare separation because of an omitted "
                "reachable path."
            ),
            (
                "Potential savings count the separated position itself "
                "because production could resolve the tail before running "
                "MCTS there."
            ),
            (
                "Exact tails maximize each player's independent remaining "
                "score under the real normal-move and grass-capture rules."
            ),
            "A capped solve would fall back to normal self-play in production.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
