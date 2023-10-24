from google.protobuf.message import DecodeError
import io
import multiprocessing as mp
import os
import numpy as np
import requests
import time
import torch
from typing import Optional
from pyhexz import hexz_pb2

from pyhexz.board import Board
from pyhexz.config import WorkerConfig
from pyhexz.errors import HexzError
from pyhexz.hexz import HexzNeuralNetwork, NeuralMCTS
import pyhexz.timing as tm


def np_tobytes(X: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, X)
    return b.getvalue()


def fetch_model(training_server_url, model_name: str = None, checkpoint: int = None) -> (HexzNeuralNetwork, str, int):
    if model_name is None:
        resp = requests.get(training_server_url + "/models/current")
        if not resp.ok:
            raise HexzError(
                f"Failed to get model info from {training_server_url}: {resp.status_code}"
            )
        j = resp.json()
        model_name = j["model_name"]
        checkpoint = j["checkpoint"]
        print(
            f"Server at {training_server_url} is using model {model_name}:{checkpoint}."
        )
    resp = requests.get(training_server_url + f"/models/{model_name}/checkpoints/{checkpoint}")
    if not resp.ok:
        raise HexzError(f"Cannot fetch model {model_name}:{checkpoint} from training server: {resp.status_code}")
    model = HexzNeuralNetwork()
    model.load_state_dict(torch.load(io.BytesIO(resp.content), map_location="cpu"))
    return model, model_name, checkpoint


def record_examples(config: WorkerConfig, progress_queue: Optional[mp.Queue] = None) -> None:
    tm.clear_perf_stats()
    worker_id = os.getpid()
    print(f"Worker {worker_id} started.")
    started = time.time()
    num_games = 0
    device = config.device
    model, model_name, checkpoint = fetch_model(config.training_server_url)
    print(
        f"Server at {config.training_server_url} is using model {model_name}:{checkpoint}."
    )
    # model = torch.compile(model)
    while time.time() - started < config.max_seconds:
        b = Board()
        m = NeuralMCTS(b, model, device=device)
        examples = m.play_game(
            runs_per_move=config.runs_per_move, progress_queue=progress_queue
        )
        num_games += 1
        if progress_queue:
            progress_queue.put(
                {
                    "games": 1,
                    "done": False,
                }
            )
        req = hexz_pb2.AddTrainingExamplesRequest(
            model_key=hexz_pb2.ModelKey(name=model_name, checkpoint=checkpoint)
        )
        unix_micros = time.time_ns() // 1000
        for ex in examples:
            req.examples.append(
                hexz_pb2.TrainingExample(
                    unix_micros=unix_micros,
                    board=np_tobytes(ex.board),
                    move_probs=np_tobytes(ex.move_probs),
                    result=ex.result,
                )
            )
        data = req.SerializeToString()
        for i in range(100):
            # Try to POST our examples to the training server in an exponential backoff loop.
            http_resp = requests.post(
                config.training_server_url + "/examples", data=data
            )
            if not http_resp.ok:
                raise HexzError(
                    f"Failed to send examples to server: {http_resp.status_code}"
                )
            try:
                resp = hexz_pb2.AddTrainingExamplesResponse.FromString(
                    http_resp.content
                )
            except DecodeError as e:
                raise HexzError(
                    f"Server replied with invalid AddTrainingExamplesResponse: {e}"
                )
            if resp.status != hexz_pb2.AddTrainingExamplesResponse.REJECTED_TRAINING:
                break
            # Server is training. Back off exponentially (with 10% jitter) and try again later.
            backoff_secs = (1.5**i) * np.random.uniform(0.90, 1.10)
            print(f"Server is training. Backing off for {backoff_secs:.2f}s (#{i})")
            time.sleep(backoff_secs)
        if resp.status == hexz_pb2.AddTrainingExamplesResponse.REJECTED_WRONG_MODEL:
            # Load newer model
            model_name = resp.latest_model.name
            checkpoint = resp.latest_model.checkpoint
            model, _, _ = fetch_model(config.training_server_url, model_name, checkpoint)
            print(f"Using new model {model_name}:{checkpoint}")
        elif resp.status != hexz_pb2.AddTrainingExamplesResponse.ACCEPTED:
            raise HexzError(
                f"Unexpected return code from training server: {hexz_pb2.AddTrainingExamplesResponse.Status.Name(resp.status)}"
            )

    if progress_queue:
        progress_queue.put(
            {
                "done": True,
            }
        )
    tm.print_perf_stats()


def main():
    config = WorkerConfig.from_env()
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"torch version: {torch.__version__}")
    if config.device == "cuda" and not torch.cuda.is_available():
        print("Device cuda not available, falling back to cpu.")
        config.device = "cpu"
    elif config.device == "mps" and not torch.backends.mps.is_available():
        print("Device mps not available, falling back to cpu.")
        config.device = "cpu"
    print(f"Running with {config=}")

    record_examples(config, progress_queue=None)


if __name__ == "__main__":
    main()
