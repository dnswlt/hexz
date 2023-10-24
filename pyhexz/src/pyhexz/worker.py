import collections
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
from pyhexz.errors import HexzError
from pyhexz.hexz import NeuralMCTS
from pyhexz.modelserver import LocalModelRepository, ModelRepository
import pyhexz.timing as tm


Config = collections.namedtuple(
    "Config",
    [
        "device",
        "max_seconds",
        "runs_per_move",
        "model_repo_base_dir",
        "training_server_url",
    ],
)


def np_tobytes(X: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, X)
    return b.getvalue()


def record_examples(
    repo: ModelRepository, config: Config, progress_queue: Optional[mp.Queue] = None
):
    tm.clear_perf_stats()
    worker_id = os.getpid()
    print(f"Worker {worker_id} started.")
    started = time.time()
    num_games = 0
    device = config.device
    resp = requests.get(config.training_server_url + "/models")
    if not resp.ok:
        raise HexzError(f"Failed to get model info from {config.training_server_url}: {resp.status_code}")
    j = resp.json()
    model_name = j["model_name"]
    checkpoint = j["checkpoint"]
    print(f"Server at {config.training_server_url} is using model {model_name}:{checkpoint}.")
    model = repo.get_model(model_name, checkpoint)
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
        http_resp = requests.post(config.training_server_url + "/examples", data=req.SerializeToString())
        if not http_resp.ok:
            raise HexzError(f"Failed to send examples to server: {http_resp.status_code}")
        try:
            resp = hexz_pb2.AddTrainingExamplesResponse.FromString(http_resp.content)
        except DecodeError as e:
            raise HexzError(f"Server replied with invalid AddTrainingExamplesResponse: {e}")
        if resp.status != hexz_pb2.AddTrainingExamplesResponse.ACCEPTED:
            print(f"Server refused our examples: {resp.error_message=}")
        if resp.HasField("latest_model"):
            if model_name != resp.latest_model.name:
                raise HexzError(f"Server started using a different model: {model_name} => {resp.latest_model.name}")
            if checkpoint != resp.latest_model.checkpoint:
                print(f"Server started using a different checkpoint: {checkpoint} => {resp.latest_model.checkpoint}")
                checkpoint = resp.latest_model.checkpoint
        

    if progress_queue:
        progress_queue.put(
            {
                "done": True,
            }
        )
    tm.print_perf_stats()


def main():
    config = Config(
        device=os.getenv("PYTORCH_DEVICE") or "cpu",
        max_seconds=int(os.getenv("HEXZ_MAX_SECONDS") or "60"),
        runs_per_move=int(os.getenv("HEXZ_RUNS_PER_MOVE") or "800"),
        training_server_url=os.getenv("HEXZ_TRAINING_SERVER_URL"),
        model_repo_base_dir=os.getenv("HEXZ_MODEL_REPO_BASE_DIR"),
    )
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
    
    if not config.model_repo_base_dir:
        raise HexzError("No model_repo_base_dir given. Did you forget to set HEXZ_MODEL_REPO_BASE_DIR?")
    repo = LocalModelRepository(config.model_repo_base_dir)
    record_examples(repo, config, progress_queue=None)


if __name__ == "__main__":
    main()
