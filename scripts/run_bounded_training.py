#!/usr/bin/env python3
"""Run or resume a self-play experiment up to a durable checkpoint limit."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_checkpoint(repo: Path, model: str) -> int:
    checkpoints = repo / "models" / "flagz" / model / "checkpoints"
    values = [int(path.name) for path in checkpoints.iterdir() if path.name.isdigit()]
    if not values:
        raise ValueError(f"No checkpoints found for {model!r}")
    return max(values)


def fetch_status(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except (URLError, TimeoutError, json.JSONDecodeError):
        return None


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def stop_container(name: str) -> None:
    if container_running(name):
        subprocess.run(
            ["docker", "stop", "--time", "30", name],
            check=False,
        )


def stop_process(process: subprocess.Popen | None, timeout: float = 40) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def stop_process_group(pid: int | None, timeout: float = 40) -> None:
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path.home() / "data" / "hexz-models"))
    parser.add_argument("--model", default="res10-rich-v1-r4")
    parser.add_argument("--max-checkpoint", type=int, default=60)
    parser.add_argument("--max-runtime-seconds", type=int, default=43_200)
    parser.add_argument("--learning-rate", default="0.0001")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--status-url", default="http://localhost:8080/training/status")
    parser.add_argument(
        "--training-params",
        default="scripts/training_params_rich_v1.json",
    )
    parser.add_argument(
        "--container-name", default="hexz-rich-v1-r4-worker"
    )
    args = parser.parse_args()

    if args.max_checkpoint <= 0 or args.max_runtime_seconds <= 0:
        parser.error("max-checkpoint and max-runtime-seconds must be positive")

    root = Path(__file__).resolve().parent.parent
    repo = Path(args.repo).resolve()
    params = (root / args.training_params).resolve()
    model_base = repo / "models" / "flagz" / args.model
    manifest = model_base / "experiment.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest}")
    if not params.exists():
        raise FileNotFoundError(f"Missing training parameters: {params}")

    current_checkpoint = latest_checkpoint(repo, args.model)
    state_path = model_base / "bounded_run.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["model"] != args.model:
            raise ValueError("Existing bounded-run state belongs to another model")
        previous_limit = state["max_checkpoint"]
        if args.max_checkpoint != previous_limit:
            if args.max_checkpoint < previous_limit:
                raise ValueError("Cannot lower an existing bounded-run limit")
            if current_checkpoint < previous_limit:
                raise ValueError(
                    "Cannot extend the bounded-run limit before the existing "
                    "limit has been reached"
                )
            state.setdefault("limit_extensions", []).append(
                {
                    "at": utcnow(),
                    "from": previous_limit,
                    "to": args.max_checkpoint,
                    "checkpoint": current_checkpoint,
                }
            )
            state["max_checkpoint"] = args.max_checkpoint
    else:
        state = {
            "model": args.model,
            "max_checkpoint": args.max_checkpoint,
            "created_at": utcnow(),
            "invocations": [],
        }

    invocation = {
        "started_at": utcnow(),
        "starting_checkpoint": current_checkpoint,
        "pid": os.getpid(),
    }
    state["invocations"].append(invocation)
    atomic_write_json(state_path, state)

    if invocation["starting_checkpoint"] >= args.max_checkpoint:
        invocation.update(
            stopped_at=utcnow(),
            reason="checkpoint_limit_already_reached",
            final_checkpoint=invocation["starting_checkpoint"],
        )
        atomic_write_json(state_path, state)
        print(
            f"{args.model} already reached checkpoint "
            f"{invocation['starting_checkpoint']}"
        )
        return 0

    interrupted = False

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    server_process = None
    attached_server_pid = None
    worker_process = None
    reason = "unknown"
    last_status = None
    started = time.monotonic()
    try:
        status = fetch_status(args.status_url)
        if status is None:
            server_env = os.environ.copy()
            server_env.update(
                {
                    "HEXZ_MODEL_NAME": args.model,
                    "HEXZ_MODEL_REPRESENTATION": "rich_v1",
                    "HEXZ_LEARNING_RATE": args.learning_rate,
                    "HEXZ_TRAINING_MAX_CHECKPOINT": str(args.max_checkpoint),
                    "HEXZ_TRAINING_PARAMS_FILE": str(params),
                }
            )
            server_process = subprocess.Popen(
                ["bash", "scripts/training_server_local.sh"],
                cwd=root,
                env=server_env,
                start_new_session=True,
            )
            invocation["server_pid"] = server_process.pid
            atomic_write_json(state_path, state)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and not interrupted:
                if server_process.poll() is not None:
                    raise RuntimeError(
                        f"Training server exited with {server_process.returncode}"
                    )
                status = fetch_status(args.status_url)
                if status is not None:
                    break
                time.sleep(1)
            if status is None:
                raise TimeoutError("Training server did not become healthy")
        else:
            # A previous runner may have died while leaving the healthy server
            # and named worker alive. Retain its process-group id so this
            # resumed invocation can still clean it up at the boundary.
            for previous in reversed(state["invocations"][:-1]):
                pid = previous.get("server_pid")
                if pid is None:
                    continue
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                attached_server_pid = pid
                invocation["attached_server_pid"] = pid
                break

        if status["model"] != args.model:
            raise RuntimeError(
                f"Port 8080 serves {status['model']!r}, expected {args.model!r}"
            )
        if status["max_checkpoint"] != args.max_checkpoint:
            raise RuntimeError(
                "Training server checkpoint limit does not match the runner"
            )

        if not container_running(args.container_name):
            worker_env = os.environ.copy()
            worker_env["HEXZ_WORKER_CONTAINER_NAME"] = args.container_name
            worker_process = subprocess.Popen(
                [
                    "bash",
                    "scripts/worker_docker_cuda.sh",
                    "host.docker.internal:50051",
                    str(args.max_runtime_seconds),
                ],
                cwd=root,
                env=worker_env,
                start_new_session=True,
            )

        while True:
            if interrupted:
                reason = "interrupted"
                break
            if time.monotonic() - started >= args.max_runtime_seconds:
                reason = "runtime_limit"
                break
            if server_process is not None and server_process.poll() is not None:
                reason = f"server_exit_{server_process.returncode}"
                break
            if worker_process is not None and worker_process.poll() is not None:
                reason = f"worker_exit_{worker_process.returncode}"
                break

            status = fetch_status(args.status_url)
            if status is not None:
                last_status = status
                invocation["last_status"] = status
                atomic_write_json(state_path, state)
                print(
                    f"checkpoint={status['checkpoint']} examples={status['examples']} "
                    f"training={status['is_training']}",
                    flush=True,
                )
                if status["checkpoint"] >= args.max_checkpoint:
                    reason = "checkpoint_limit_reached"
                    break
            time.sleep(args.poll_seconds)
    finally:
        # Stop the named container first. At the checkpoint boundary it is
        # deliberately suspended by the server and cannot upload more data.
        stop_container(args.container_name)
        stop_process(worker_process)
        stop_process(server_process)
        if server_process is None:
            stop_process_group(attached_server_pid)
        final_checkpoint = latest_checkpoint(repo, args.model)
        invocation.update(
            stopped_at=utcnow(),
            reason=reason,
            final_checkpoint=final_checkpoint,
        )
        if last_status is not None:
            invocation["last_status"] = last_status
        atomic_write_json(state_path, state)

    print(f"Stopped: {reason}; latest checkpoint={final_checkpoint}")
    return 0 if reason == "checkpoint_limit_reached" else 2


if __name__ == "__main__":
    raise SystemExit(main())
