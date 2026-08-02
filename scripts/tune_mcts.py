#!/usr/bin/env python3
"""Tune checkpoint-50 PUCT/FPU settings with staged paired arenas."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


Z95 = 1.959963984540054


@dataclass(frozen=True)
class SearchConfig:
    uct_c: float
    initial_root_q_value: float
    initial_q_penalty: float

    @property
    def label(self) -> str:
        def fmt(value: float) -> str:
            return f"{value:g}".replace("-", "m").replace(".", "p")

        return (
            f"uct{fmt(self.uct_c)}-root{fmt(self.initial_root_q_value)}-"
            f"pen{fmt(self.initial_q_penalty)}"
        )


HISTORICAL = SearchConfig(1.5, 0.0, 0.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(repo_root: Path) -> str:
    """Hash the exact tuning/search sources, including uncommitted files."""
    digest = hashlib.sha256()
    for relative in (
        "cpp/base.h",
        "cpp/cpuserver.cc",
        "cpp/cpuserver.h",
        "cpp/cpuserver_main.cc",
        "cpp/mcts.cc",
        "cpp/mcts.h",
        "cmd/nbench2/main.go",
        "cmd/nbench2/concurrent.go",
        "scripts/nbench2.sh",
        "scripts/tune_mcts.py",
    ):
        path = repo_root / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def game_score(game: dict, candidate_is_p1: bool) -> float:
    winner = int(game.get("winner", 0))
    if winner == 0:
        return 0.5
    return 1.0 if (winner == 1) == candidate_is_p1 else 0.0


def summarize(stats_file: Path) -> dict:
    records = [json.loads(line) for line in stats_file.read_text().splitlines()]
    if len(records) != 2:
        raise RuntimeError(f"Expected two benchmark records in {stats_file}, got {len(records)}")
    first, reverse = records
    first_by_id = {g["positionId"]: g for g in first["gameResults"]}
    reverse_by_id = {g["positionId"]: g for g in reverse["gameResults"]}
    if first_by_id.keys() != reverse_by_id.keys():
        raise RuntimeError(f"Seat-swapped position sets differ in {stats_file}")

    pair_scores = [
        (game_score(first_by_id[k], True) + game_score(reverse_by_id[k], False)) / 2
        for k in sorted(first_by_id)
    ]
    score = sum(pair_scores) / len(pair_scores)
    if len(pair_scores) > 1:
        variance = sum((x - score) ** 2 for x in pair_scores) / (len(pair_scores) - 1)
        margin = Z95 * math.sqrt(variance / len(pair_scores))
    else:
        margin = 0.0

    candidate_wins = sum(g.get("winner", 0) == 1 for g in first_by_id.values())
    candidate_wins += sum(g.get("winner", 0) == 2 for g in reverse_by_id.values())
    baseline_wins = sum(g.get("winner", 0) == 2 for g in first_by_id.values())
    baseline_wins += sum(g.get("winner", 0) == 1 for g in reverse_by_id.values())
    games = 2 * len(pair_scores)
    return {
        "positions": len(pair_scores),
        "games": games,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "draws": games - candidate_wins - baseline_wins,
        "score": score,
        "score_ci95": [max(0.0, score - margin), min(1.0, score + margin)],
        "duration_seconds": sum(float(r["durationSeconds"]) for r in records),
    }


def minimal_change(config: SearchConfig) -> float:
    return (
        abs(config.uct_c - HISTORICAL.uct_c)
        + abs(config.initial_root_q_value)
        + abs(config.initial_q_penalty)
    )


def choose_screen_winner(results: list[dict]) -> SearchConfig:
    eligible = [r for r in results if r["summary"]["score"] > 0.5]
    if not eligible:
        return HISTORICAL
    winner = max(
        eligible,
        key=lambda r: (r["summary"]["score"], -minimal_change(r["config_object"])),
    )
    return winner["config_object"]


class Campaign:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.started_monotonic = time.monotonic()
        self.deadline = self.started_monotonic + args.budget_minutes * 60
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest: dict = {
            "kind": "mcts_search_tuning",
            "status": "running",
            "started": datetime.now(timezone.utc).isoformat(),
            "model": {"name": args.model, "checkpoint": args.checkpoint},
            "iterations": args.iterations,
            "budget_minutes": args.budget_minutes,
            "historical_inference_config": asdict(HISTORICAL),
            "matches": [],
        }
        self.write_manifest()

    def write_manifest(self) -> None:
        serializable = {
            key: value for key, value in self.manifest.items() if key != "matches"
        }
        serializable["matches"] = [
            {key: value for key, value in match.items() if key != "config_object"}
            for match in self.manifest.get("matches", [])
        ]
        self.manifest_path.write_text(json.dumps(serializable, indent=2) + "\n")

    def remaining_seconds(self) -> float:
        return self.deadline - time.monotonic()

    def cleanup(self) -> None:
        subprocess.run(
            ["docker", "stop", "hexz-nbench2-p1", "hexz-nbench2-p2"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def run_match(self, stage: str, index: int, config: SearchConfig, offset: int, games: int) -> dict:
        remaining = self.remaining_seconds()
        if remaining <= 0:
            raise TimeoutError("Tuning wall-time budget exhausted")
        stats_file = self.output_dir / f"{stage}-{index:02d}-{config.label}.jsonl"
        env = os.environ.copy()
        env.update(
            {
                "HEXZ_CPUSERVER_IMAGE": self.args.image,
                "HEXZ_MODEL_REPO_BASE_DIR": str(self.args.repo),
                "HEXZ_NBENCH_GAMES": str(games),
                "HEXZ_NBENCH_POSITION_OFFSET": str(offset),
                "HEXZ_NBENCH_ITERATIONS": str(self.args.iterations),
                "HEXZ_NBENCH_CONCURRENCY": str(min(self.args.concurrency, games)),
                "HEXZ_NBENCH_POSITIONS_FILE": str(self.args.positions_file),
                "HEXZ_NBENCH_STATS_FILE": str(stats_file),
                "HEXZ_NBENCH_STARTUP_SECONDS": "2",
                "HEXZ_NBENCH_P1_UCT_C": str(config.uct_c),
                "HEXZ_NBENCH_P1_INITIAL_ROOT_Q_VALUE": str(config.initial_root_q_value),
                "HEXZ_NBENCH_P1_INITIAL_Q_PENALTY": str(config.initial_q_penalty),
                "HEXZ_NBENCH_P2_UCT_C": str(HISTORICAL.uct_c),
                "HEXZ_NBENCH_P2_INITIAL_ROOT_Q_VALUE": str(HISTORICAL.initial_root_q_value),
                "HEXZ_NBENCH_P2_INITIAL_Q_PENALTY": str(HISTORICAL.initial_q_penalty),
            }
        )
        model_key = f"{self.args.model}:{self.args.checkpoint}"
        print(
            f"[{stage} {index}] {config.label} vs {HISTORICAL.label}; "
            f"positions {offset}..{offset + games - 1}",
            flush=True,
        )
        try:
            subprocess.run(
                ["bash", "scripts/nbench2.sh", model_key, model_key],
                cwd=self.args.repo_root,
                env=env,
                check=True,
                timeout=remaining,
            )
        finally:
            self.cleanup()
        summary = summarize(stats_file)
        print(
            f"  score={summary['score']:.3f}, "
            f"CI={summary['score_ci95'][0]:.3f}..{summary['score_ci95'][1]:.3f}",
            flush=True,
        )
        result = {
            "stage": stage,
            "config": asdict(config),
            "config_object": config,
            "baseline": asdict(HISTORICAL),
            "position_offset": offset,
            "positions": games,
            "stats_file": str(stats_file),
            "summary": summary,
        }
        self.manifest["matches"].append(result)
        self.write_manifest()
        return result

    def run(self) -> None:
        model_path = (
            self.args.repo
            / "models"
            / "flagz"
            / self.args.model
            / "checkpoints"
            / str(self.args.checkpoint)
            / "scriptmodule.pt"
        )
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        image_id = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.args.image],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.args.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.manifest["artifacts"] = {
            "model_path": str(model_path),
            "model_sha256": sha256(model_path),
            "positions_file": str(self.args.positions_file),
            "positions_sha256": sha256(self.args.positions_file),
            "cpuserver_image": self.args.image,
            "cpuserver_image_id": image_id,
            "git_commit": commit,
            "source_fingerprint_sha256": source_fingerprint(self.args.repo_root),
        }
        self.write_manifest()

        fpu_candidates = [
            SearchConfig(1.5, -0.2, 0.3),
            SearchConfig(1.5, -0.1, 0.15),
            SearchConfig(1.5, -0.2, 0.0),
            SearchConfig(1.5, 0.0, 0.3),
        ]
        fpu_results = [
            self.run_match("fpu", i, config, offset=0, games=24)
            for i, config in enumerate(fpu_candidates, 1)
        ]
        selected_fpu = choose_screen_winner(fpu_results)
        self.manifest["selected_fpu"] = asdict(selected_fpu)
        self.write_manifest()

        puct_candidates = [
            SearchConfig(c, selected_fpu.initial_root_q_value, selected_fpu.initial_q_penalty)
            for c in (1.0, 1.5, 2.0)
        ]
        puct_results = [
            self.run_match("puct", i, config, offset=24, games=24)
            for i, config in enumerate(puct_candidates, 1)
        ]
        finalist = choose_screen_winner(puct_results)
        self.manifest["finalist"] = asdict(finalist)
        self.write_manifest()

        final = self.run_match("confirmation", 1, finalist, offset=48, games=80)
        adopted = final["summary"]["score_ci95"][0] > 0.5
        selected = finalist if adopted else HISTORICAL
        self.manifest.update(
            {
                "status": "complete",
                "done": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - self.started_monotonic,
                "adopted_finalist": adopted,
                "selected_shared_config": asdict(selected),
            }
        )
        self.write_manifest()
        print(f"Selected shared configuration: {selected}", flush=True)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "data" / "hexz-models")
    parser.add_argument("--model", default="res10-r4-cp62")
    parser.add_argument("--checkpoint", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--budget-minutes", type=float, default=45)
    parser.add_argument("--image", default="hexz-cpuserver-cuda:latest")
    parser.add_argument(
        "--positions-file",
        type=Path,
        default=repo_root / "testdata/nbench/flagz_initial_v1.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "log"
        / f"mcts-tuning-cp50-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    args = parser.parse_args()
    args.repo_root = repo_root
    if args.iterations <= 0 or args.concurrency <= 0 or args.budget_minutes <= 0:
        parser.error("iterations, concurrency, and budget must be positive")
    return args


def main() -> None:
    args = parse_args()
    campaign = Campaign(args)
    try:
        campaign.run()
    except Exception as error:
        campaign.cleanup()
        campaign.manifest.update(
            {
                "status": "failed",
                "done": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        campaign.write_manifest()
        raise


if __name__ == "__main__":
    main()
