#!/usr/bin/env python3
"""Deterministically re-search a phase-balanced sample of archived positions.

The campaign compares the recorded self-play policy with fresh, noise-free
searches at three depths. It is read-only with respect to the model repository:
all manifests, selected positions, teacher policies, and summaries are written
below the local output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import io
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import grpc
import numpy as np
import torch

from pyhexz import hexz_pb2, hexz_pb2_grpc


PHASES = (
    ("moves_00_05", 0, 6),
    ("moves_06_11", 6, 12),
    ("moves_12_23", 12, 24),
    ("moves_24_47", 24, 48),
    ("moves_48_63", 48, 64),
    ("moves_64_plus", 64, math.inf),
)
PHASE_GROUPS = {
    "moves_00_11": lambda move: move < 12,
    "moves_12_plus": lambda move: move >= 12,
}
CHANNEL_SWAP = (5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10)
POLICY_SIZE = 2 * 11 * 10
CONTAINER = "hexz-research-archived"


@dataclass(frozen=True)
class ArchivedPosition:
    position_id: str
    phase: str
    archive_relpath: str
    game_id: str
    example: hexz_pb2.TrainingExample
    state: hexz_pb2.GameEngineState
    recorded_policy: np.ndarray


def phase_name(move: int) -> str:
    for name, start, stop in PHASES:
        if start <= move < stop:
            return name
    raise ValueError(f"invalid move number: {move}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(repo_root: Path) -> str:
    paths = (
        repo_root / "scripts" / "research_archived_positions.py",
        repo_root / "cpp" / "base.h",
        repo_root / "cpp" / "board.cc",
        repo_root / "cpp" / "cpuserver.cc",
        repo_root / "cpp" / "mcts.cc",
        repo_root / "proto" / "hexz.proto",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repo_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_from_relpath(relpath: str) -> int | None:
    parts = Path(relpath).parts
    if len(parts) < 4 or parts[0] != "checkpoints":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def parse_checkpoint_set(value: str) -> set[int]:
    checkpoints: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            start_text, stop_text = part.split("-", maxsplit=1)
            start, stop = int(start_text), int(stop_text)
            if start > stop:
                raise argparse.ArgumentTypeError(f"descending range: {part}")
            checkpoints.update(range(start, stop + 1))
        else:
            checkpoints.add(int(part))
    if not checkpoints or min(checkpoints) < 0:
        raise argparse.ArgumentTypeError("archive checkpoints must be non-negative")
    return checkpoints


def decode_tensor(data: bytes, encoding: int) -> np.ndarray:
    if encoding == hexz_pb2.TrainingExample.PYTORCH:
        value = torch.load(io.BytesIO(data), weights_only=True).numpy()
    elif encoding == hexz_pb2.TrainingExample.NUMPY:
        value = np.load(io.BytesIO(data))
    else:
        raise ValueError(f"unsupported training-example encoding: {encoding}")
    return np.asarray(value, dtype=np.float64)


def decode_policy(example: hexz_pb2.TrainingExample) -> np.ndarray:
    policy = decode_tensor(example.move_probs, example.encoding).reshape(-1)
    if policy.size != POLICY_SIZE or not np.isfinite(policy).all():
        raise ValueError("invalid archived policy")
    total = float(policy.sum())
    if total <= 0 or not math.isclose(total, 1.0, abs_tol=1e-4):
        raise ValueError(f"invalid archived policy sum: {total}")
    return policy / total


def example_to_state(
    example: hexz_pb2.TrainingExample,
) -> hexz_pb2.GameEngineState:
    """Invert Board::Tensor for the fields consumed by Board::FromProto."""
    board_tensor = decode_tensor(example.board, example.encoding)
    if board_tensor.shape != (11, 11, 10):
        raise ValueError(f"invalid archived board shape: {board_tensor.shape}")
    if example.turn == 1:
        board_tensor = board_tensor[list(CHANNEL_SWAP)]
    elif example.turn != 0:
        raise ValueError(f"invalid archived turn: {example.turn}")

    board = hexz_pb2.Board(
        turn=example.turn + 1,
        move=example.move.move,
        state=hexz_pb2.Board.RUNNING,
    )
    board.score.extend(
        [
            int(np.rint(board_tensor[1].sum())),
            int(np.rint(board_tensor[6].sum())),
        ]
    )
    for player in (0, 1):
        pieces = [-1] + [0] * 7
        pieces[hexz_pb2.Field.FLAG] = int(
            np.rint(board_tensor[4 + 5 * player, 0, 0])
        )
        board.resources.add().num_pieces.extend(pieces)

    free_cells = 0
    normal_moves = [0, 0]
    for row in range(11):
        for col in range(10 - row % 2):
            field = board.flat_fields.add()
            flag_owner = next(
                (
                    player
                    for player in (0, 1)
                    if board_tensor[5 * player, row, col] > 0.5
                ),
                None,
            )
            value_owner = next(
                (
                    player
                    for player in (0, 1)
                    if board_tensor[1 + 5 * player, row, col] > 0
                ),
                None,
            )
            grass = int(np.rint(board_tensor[10, row, col]))
            if flag_owner is not None:
                field.type = hexz_pb2.Field.FLAG
                field.owner = flag_owner + 1
            elif value_owner is not None:
                field.type = hexz_pb2.Field.NORMAL
                field.owner = value_owner + 1
                field.value = int(
                    np.rint(board_tensor[1 + 5 * value_owner, row, col])
                )
            elif grass > 0:
                field.type = hexz_pb2.Field.GRASS
                field.value = grass
            else:
                field.type = hexz_pb2.Field.NORMAL
                blocked = 0
                next_values = []
                for player in (0, 1):
                    if board_tensor[2 + 5 * player, row, col] > 0.5:
                        blocked |= 1 << player
                    next_value = int(
                        np.rint(board_tensor[3 + 5 * player, row, col])
                    )
                    next_values.append(next_value)
                    normal_moves[player] += int(next_value > 0)
                field.blocked = blocked
                field.next_val.extend(next_values)
                free_cells += int(blocked != 3)

    flagz = hexz_pb2.GameEngineFlagzState(
        board=board,
        free_cells=free_cells,
        normal_moves=normal_moves,
    )
    return hexz_pb2.GameEngineState(flagz=flagz)


def position_id(state: hexz_pb2.GameEngineState) -> str:
    data = state.SerializeToString(deterministic=True)
    return hashlib.sha256(data).hexdigest()[:16]


def select_positions(
    model_base: Path,
    checkpoints: int | set[int],
    per_phase: int,
    seed: int,
    min_move: int = 0,
    one_per_game_phase: bool = False,
) -> tuple[list[ArchivedPosition], dict[str, int]]:
    if isinstance(checkpoints, int):
        checkpoints = {checkpoints}
    index_path = model_base / "index" / "requests.jsonl"
    rng = random.Random(seed)
    selected_phases = {
        name for name, _, stop in PHASES if stop > min_move
    }
    reservoirs: dict[str, list[ArchivedPosition]] = {
        name: [] for name, _, _ in PHASES if name in selected_phases
    }
    seen = {name: 0 for name in selected_phases}
    archive_count = 0
    with index_path.open(encoding="utf-8") as index:
        for line in index:
            if not line.strip():
                continue
            entry = json.loads(line)
            if checkpoint_from_relpath(entry["relpath"]) not in checkpoints:
                continue
            archive_count += 1
            with gzip.open(model_base / entry["relpath"], "rb") as source:
                request = hexz_pb2.AddTrainingExamplesRequest()
                request.ParseFromString(source.read())
            eligible = [
                example
                for example in request.examples
                if example.model_key.checkpoint in checkpoints
                and example.move.move >= min_move
            ]
            if one_per_game_phase:
                by_phase: dict[str, list[hexz_pb2.TrainingExample]] = {
                    phase: [] for phase in selected_phases
                }
                for example in eligible:
                    by_phase[phase_name(example.move.move)].append(example)
                eligible = [
                    rng.choice(by_phase[phase])
                    for phase, _, _ in PHASES
                    if phase in selected_phases and by_phase[phase]
                ]
            for example in eligible:
                phase = phase_name(example.move.move)
                if phase not in selected_phases:
                    continue
                state = example_to_state(example)
                selected = ArchivedPosition(
                    position_id=position_id(state),
                    phase=phase,
                    archive_relpath=entry["relpath"],
                    game_id=request.game_id,
                    example=hexz_pb2.TrainingExample.FromString(
                        example.SerializeToString()
                    ),
                    state=state,
                    recorded_policy=decode_policy(example),
                )
                seen[phase] += 1
                reservoir = reservoirs[phase]
                if len(reservoir) < per_phase:
                    reservoir.append(selected)
                else:
                    replacement = rng.randrange(seen[phase])
                    if replacement < per_phase:
                        reservoir[replacement] = selected
    missing = {
        phase: len(values)
        for phase, values in reservoirs.items()
        if len(values) < per_phase
    }
    if missing:
        raise ValueError(
            f"not enough positions for {per_phase} per phase: {missing}"
        )
    positions = [
        position
        for phase, _, _ in PHASES
        if phase in selected_phases
        for position in reservoirs[phase]
    ]
    if len({position.position_id for position in positions}) != len(positions):
        raise ValueError("sample contains duplicate reconstructed states")
    return positions, {"archives": archive_count, **seen}


def response_policy(response: hexz_pb2.SuggestMoveResponse) -> np.ndarray:
    policy = np.zeros(POLICY_SIZE, dtype=np.float64)
    for move in response.move_stats.moves:
        if move.type == hexz_pb2.Field.FLAG:
            plane = 0
        elif move.type == hexz_pb2.Field.NORMAL:
            plane = 1
        else:
            raise ValueError(f"unexpected response move type: {move.type}")
        index = plane * 110 + move.row * 10 + move.col
        final = [
            score.score
            for score in move.scores
            if score.kind == hexz_pb2.SuggestMoveStats.FINAL
        ]
        if len(final) != 1:
            raise ValueError("response move does not have exactly one final score")
        policy[index] = final[0]
    total = float(policy.sum())
    if total <= 0 or not np.isfinite(policy).all():
        raise ValueError(f"invalid response policy sum: {total}")
    return policy / total


async def research_depth(
    stub: hexz_pb2_grpc.CPUPlayerServiceStub,
    positions: list[ArchivedPosition],
    iterations: int,
    concurrency: int,
    rpc_timeout: float,
) -> tuple[list[np.ndarray], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(position: ArchivedPosition) -> np.ndarray:
        async with semaphore:
            response = await stub.SuggestMove(
                hexz_pb2.SuggestMoveRequest(
                    max_iterations=iterations,
                    game_engine_state=position.state,
                ),
                timeout=rpc_timeout,
            )
            return response_policy(response)

    started = time.monotonic()
    policies = await asyncio.gather(*(one(position) for position in positions))
    return policies, time.monotonic() - started


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def kl(source: np.ndarray) -> float:
        mask = source > 0
        return float(np.sum(source[mask] * np.log(source[mask] / midpoint[mask])))

    return 0.5 * (kl(left) + kl(right))


def comparison_rows(
    left: list[np.ndarray], right: list[np.ndarray]
) -> list[dict[str, float | bool]]:
    return [
        {
            "total_variation": float(0.5 * np.abs(a - b).sum()),
            "js_divergence_nats": js_divergence(a, b),
            "top1_same": bool(int(a.argmax()) == int(b.argmax())),
        }
        for a, b in zip(left, right, strict=True)
    ]


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - radius, center + radius]


def summarize_rows(rows: list[dict], seed: int) -> dict:
    tv = np.asarray([row["total_variation"] for row in rows])
    js = np.asarray([row["js_divergence_nats"] for row in rows])
    changes = sum(not row["top1_same"] for row in rows)
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(
        rng.choice(tv, size=(5000, len(tv)), replace=True), axis=1
    )
    return {
        "positions": len(rows),
        "mean_total_variation": float(tv.mean()),
        "mean_total_variation_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "median_total_variation": float(np.median(tv)),
        "p90_total_variation": float(np.quantile(tv, 0.9)),
        "mean_js_divergence_nats": float(js.mean()),
        "top1_changes": changes,
        "top1_change_rate": changes / len(rows),
        "top1_change_rate_ci95": wilson_interval(changes, len(rows)),
    }


def sparse_policy(policy: np.ndarray) -> list[list[float | int]]:
    return [
        [int(index), float(value)]
        for index, value in enumerate(policy)
        if value > 0
    ]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def docker_cleanup() -> None:
    subprocess.run(
        ["docker", "stop", CONTAINER],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def start_server(args: argparse.Namespace) -> None:
    docker_cleanup()
    model_path = (
        f"/models/models/flagz/{args.model}/checkpoints/"
        f"{args.checkpoint}/scriptmodule.pt"
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            CONTAINER,
            "-p",
            f"{args.port}:{args.port}",
            "-v",
            f"{args.repo}:/models:ro",
            "--gpus",
            "all",
            args.image,
            "--device=cuda",
            "--max_think_time_ms=0",
            "--uct_c=1.5",
            "--initial_root_q_value=0",
            "--initial_q_penalty=0",
            "--tail_solver_max_states=0",
            f"--model_path={model_path}",
            f"--model_key={args.model}:{args.checkpoint}",
            f"--server_addr=0.0.0.0:{args.port}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


async def run_searches(
    args: argparse.Namespace, positions: list[ArchivedPosition]
) -> tuple[dict[int, list[np.ndarray]], dict[int, float]]:
    channel = grpc.aio.insecure_channel(f"localhost:{args.port}")
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=30)
        stub = hexz_pb2_grpc.CPUPlayerServiceStub(channel)
        policies = {}
        durations = {}
        for iterations in args.iterations:
            print(
                f"Re-searching {len(positions)} positions at "
                f"{iterations:,} iterations...",
                flush=True,
            )
            policies[iterations], durations[iterations] = await research_depth(
                stub,
                positions,
                iterations,
                args.concurrency,
                args.rpc_timeout_seconds,
            )
            print(f"  completed in {durations[iterations]:.1f}s", flush=True)
        return policies, durations
    finally:
        await channel.close()


def analyze(
    positions: list[ArchivedPosition],
    policies: dict[int, list[np.ndarray]],
    seed: int,
) -> tuple[dict, list[dict]]:
    depths = list(policies)
    named = {"archive": [p.recorded_policy for p in positions]}
    named.update({str(depth): policies[depth] for depth in depths})
    if len(depths) == 1:
        result_rows = []
        for index, position in enumerate(positions):
            selected_index = (
                (
                    0
                    if position.example.move.cell_type == hexz_pb2.Field.FLAG
                    else 1
                )
                * 110
                + position.example.move.row * 10
                + position.example.move.col
            )
            result_rows.append(
                {
                    "position_id": position.position_id,
                    "phase": position.phase,
                    "archive_relpath": position.archive_relpath,
                    "game_id": position.game_id,
                    "move": position.example.move.move,
                    "turn": position.example.turn,
                    "result": position.example.result,
                    "selected_action": selected_index,
                    "policies": {
                        "archive": sparse_policy(position.recorded_policy),
                        str(depths[0]): sparse_policy(policies[depths[0]][index]),
                    },
                    "comparisons": {},
                }
            )
        return {
            "comparisons": {},
            "decision": {
                "teacher_only": True,
                "teacher_iterations": depths[0],
            },
        }, result_rows
    if len(depths) != 3:
        raise ValueError("analysis requires either one or three search depths")
    pairs = [
        ("archive", str(depths[0])),
        ("archive", str(depths[-1])),
        (str(depths[0]), str(depths[1])),
        (str(depths[1]), str(depths[2])),
        (str(depths[0]), str(depths[2])),
    ]
    comparisons = {}
    all_rows = {}
    for pair_index, (left_name, right_name) in enumerate(pairs):
        label = f"{left_name}_vs_{right_name}"
        rows = comparison_rows(named[left_name], named[right_name])
        all_rows[label] = rows
        comparisons[label] = {
            "overall": summarize_rows(rows, seed + pair_index),
            "by_phase": {},
            "by_group": {},
        }
        for phase_index, (phase, _, _) in enumerate(PHASES):
            phase_rows = [
                row
                for position, row in zip(positions, rows, strict=True)
                if position.phase == phase
            ]
            if phase_rows:
                comparisons[label]["by_phase"][phase] = summarize_rows(
                    phase_rows, seed + 100 * pair_index + phase_index
                )

        for group_index, (group, predicate) in enumerate(PHASE_GROUPS.items()):
            group_rows = [
                row
                for position, row in zip(positions, rows, strict=True)
                if predicate(position.example.move.move)
            ]
            if group_rows:
                comparisons[label]["by_group"][group] = summarize_rows(
                    group_rows, seed + 1000 * pair_index + group_index
                )

    depth_comparison = comparisons[f"{depths[0]}_vs_{depths[2]}"]
    convergence_comparison = comparisons[f"{depths[1]}_vs_{depths[2]}"]

    def classify(depth_gap: dict, convergence: dict) -> dict:
        material = (
            depth_gap["mean_total_variation"] >= 0.10
            or depth_gap["top1_change_rate"] >= 0.10
        )
        stable = (
            convergence["mean_total_variation"] <= 0.05
            and convergence["top1_change_rate"] <= 0.10
        )
        return {
            "material_depth_gap": material,
            "deep_teacher_stable": stable,
            "student_is_warranted": material and stable,
        }

    overall_decision = classify(
        depth_comparison["overall"], convergence_comparison["overall"]
    )
    post_opening_decision = classify(
        depth_comparison["by_group"]["moves_12_plus"],
        convergence_comparison["by_group"]["moves_12_plus"],
    )
    decision = {
        "criteria": {
            "material_depth_gap": (
                f"{depths[0]} vs {depths[2]} mean TV >= 0.10 or "
                "top-1 change rate >= 0.10"
            ),
            "deep_teacher_stable": (
                f"{depths[1]} vs {depths[2]} mean TV <= 0.05 and "
                "top-1 change rate <= 0.10"
            ),
        },
        "overall": overall_decision,
        "moves_12_plus": post_opening_decision,
        "isolated_student_is_warranted": post_opening_decision[
            "student_is_warranted"
        ],
        "recommended_target_mix": (
            f"retain recorded targets for moves 0-11; use the "
            f"{depths[1]}-iteration deterministic teacher for moves 12+"
            if post_opening_decision["student_is_warranted"]
            else None
        ),
    }

    result_rows = []
    for index, position in enumerate(positions):
        selected_index = (
            (0 if position.example.move.cell_type == hexz_pb2.Field.FLAG else 1)
            * 110
            + position.example.move.row * 10
            + position.example.move.col
        )
        result_rows.append(
            {
                "position_id": position.position_id,
                "phase": position.phase,
                "archive_relpath": position.archive_relpath,
                "game_id": position.game_id,
                "move": position.example.move.move,
                "turn": position.example.turn,
                "result": position.example.result,
                "selected_action": selected_index,
                "policies": {
                    "archive": sparse_policy(position.recorded_policy),
                    **{
                        str(depth): sparse_policy(policies[depth][index])
                        for depth in depths
                    },
                },
                "comparisons": {
                    label: rows[index] for label, rows in all_rows.items()
                },
            }
        )
    return {"comparisons": comparisons, "decision": decision}, result_rows


def parse_iterations(value: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("iterations must be integers") from exc
    if len(depths) not in (1, 3) or min(depths) <= 0 or list(depths) != sorted(set(depths)):
        raise argparse.ArgumentTypeError(
            "iterations must contain one positive value or three increasing positive values"
        )
    return depths


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "data" / "hexz-models")
    parser.add_argument("--model", default="res10-r4-cp62")
    parser.add_argument("--checkpoint", type=int, default=50)
    parser.add_argument(
        "--archive-checkpoints",
        type=parse_checkpoint_set,
        help="checkpoints whose archived positions may be sampled; defaults to --checkpoint",
    )
    parser.add_argument("--per-phase", type=int, default=16)
    parser.add_argument(
        "--min-move",
        type=int,
        default=0,
        help="sample only archived positions at or after this move",
    )
    parser.add_argument(
        "--one-per-game-phase",
        action="store_true",
        help="consider at most one randomly selected position per game and phase",
    )
    parser.add_argument("--iterations", type=parse_iterations, default=(800, 3200, 6400))
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--port", type=int, default=50173)
    parser.add_argument("--rpc-timeout-seconds", type=float, default=300)
    parser.add_argument("--image", default="hexz-cpuserver-cuda:latest")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "log"
        / f"archived-research-cp50-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    args = parser.parse_args()
    args.repo_root = repo_root
    if args.archive_checkpoints is None:
        args.archive_checkpoints = {args.checkpoint}
    if (
        args.checkpoint < 0
        or args.per_phase <= 0
        or args.concurrency <= 0
        or args.min_move < 0
    ):
        parser.error("checkpoint must be non-negative; sample and concurrency must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_base = args.repo / "models" / "flagz" / args.model
    model_path = model_base / "checkpoints" / str(args.checkpoint) / "scriptmodule.pt"
    index_path = model_base / "index" / "requests.jsonl"
    started = time.monotonic()
    manifest = {
        "kind": "archived_position_research",
        "status": "running",
        "started": datetime.now(timezone.utc).isoformat(),
        "model": {"name": args.model, "checkpoint": args.checkpoint},
        "sample": {
            "method": "deterministic reservoir sample, balanced across selected move phases",
            "per_phase": args.per_phase,
            "seed": args.seed,
            "min_move": args.min_move,
            "one_per_game_phase": args.one_per_game_phase,
            "archive_checkpoints": sorted(args.archive_checkpoints),
        },
        "search": {
            "iterations": list(args.iterations),
            "concurrency": args.concurrency,
            "uct_c": 1.5,
            "initial_root_q_value": 0.0,
            "initial_q_penalty": 0.0,
            "root_noise": False,
            "tail_solver": False,
        },
    }
    try:
        image_id = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.image],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=args.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest["artifacts"] = {
            "model_path": str(model_path),
            "model_sha256": sha256(model_path),
            "archive_index": str(index_path),
            "archive_index_sha256": sha256(index_path),
            "cpuserver_image": args.image,
            "cpuserver_image_id": image_id,
            "git_commit": commit,
            "source_fingerprint_sha256": source_fingerprint(args.repo_root),
        }
        positions, population = select_positions(
            model_base,
            args.archive_checkpoints,
            args.per_phase,
            args.seed,
            min_move=args.min_move,
            one_per_game_phase=args.one_per_game_phase,
        )
        manifest["sample"]["positions"] = len(positions)
        manifest["sample"]["eligible_population"] = population
        write_json(args.output_dir / "manifest.json", manifest)
        print(
            f"Selected {len(positions)} archived positions "
            f"({args.per_phase} per phase).",
            flush=True,
        )
        start_server(args)
        policies, durations = asyncio.run(run_searches(args, positions))
        analysis, result_rows = analyze(positions, policies, args.seed)
        teacher_depth = max(policies)
        teacher_examples_path = args.output_dir / "teacher_examples.npz"
        np.savez_compressed(
            teacher_examples_path,
            boards=np.stack(
                [decode_tensor(p.example.board, p.example.encoding) for p in positions]
            ).astype(np.float32),
            action_masks=np.stack(
                [
                    decode_tensor(p.example.action_mask, p.example.encoding)
                    for p in positions
                ]
            ).astype(bool),
            move_probs=np.stack(policies[teacher_depth])
            .reshape(-1, 2, 11, 10)
            .astype(np.float32),
            values=np.asarray([p.example.result for p in positions], dtype=np.float32)
            .reshape(-1, 1),
            moves=np.asarray([p.example.move.move for p in positions], dtype=np.int32),
            position_ids=np.asarray([p.position_id for p in positions]),
        )
        results_path = args.output_dir / "results.jsonl"
        with results_path.open("w", encoding="utf-8") as output:
            for row in result_rows:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
        summary = {
            **analysis,
            "search_duration_seconds": {
                str(depth): duration for depth, duration in durations.items()
            },
        }
        write_json(args.output_dir / "summary.json", summary)
        manifest.update(
            {
                "status": "complete",
                "done": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "outputs": {
                    "summary": str(args.output_dir / "summary.json"),
                    "results": str(results_path),
                    "teacher_examples": str(teacher_examples_path),
                },
                "decision": analysis["decision"],
            }
        )
        write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(analysis["decision"], indent=2), flush=True)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "done": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(args.output_dir / "manifest.json", manifest)
        raise
    finally:
        docker_cleanup()


if __name__ == "__main__":
    main()
