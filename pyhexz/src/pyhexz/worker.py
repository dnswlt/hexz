from google.protobuf import json_format
from google.protobuf.message import DecodeError
import io
import logging
import logging.config
import numpy as np
import requests
import time
import torch
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


class SelfPlayWorker:
    """SelfPlayWorker plays games for a configurable amount of time and sends the generated
    examples to a training server.
    """

    logger = logging.getLogger("SelfPlayWorker")

    def __init__(self, config: WorkerConfig):
        self.config = config

    def fetch_model(
        self, model_key: hexz_pb2.ModelKey = None
    ) -> tuple[hexz_pb2.ModelKey, HexzNeuralNetwork]:
        training_server_url = self.config.training_server_url
        if model_key is None:
            resp = requests.get(
                training_server_url + "/models/current",
                timeout=self.config.http_client_timeout,
            )
            if not resp.ok:
                raise HexzError(
                    f"Failed to get model info from {training_server_url}: {resp.status_code}"
                )
            model_key = json_format.Parse(resp.content, hexz_pb2.ModelKey())
        resp = requests.get(
            training_server_url
            + f"/models/{model_key.name}/checkpoints/{model_key.checkpoint}",
            timeout=self.config.http_client_timeout,
        )
        if not resp.ok:
            raise HexzError(
                f"Cannot fetch model {model_key} from training server: {resp.status_code}"
            )
        model = HexzNeuralNetwork()
        model.load_state_dict(torch.load(io.BytesIO(resp.content), map_location="cpu"))
        return model_key, model

    def generate_examples(self) -> None:
        config = self.config
        self.logger.info(f"Running with {config=}")
        tm.clear_perf_stats()
        started = time.time()
        num_games = 0
        device = config.device
        model_key, model = self.fetch_model()
        self.logger.info(
            f"Server at {config.training_server_url} is using model {model_key.name}:{model_key.checkpoint}."
        )
        # model = torch.compile(model)
        while time.time() - started < config.max_seconds:
            b = Board()
            m = NeuralMCTS(b, model, device=device)
            examples = m.play_game(
                runs_per_move=config.runs_per_move, progress_queue=None
            )
            num_games += 1
            req = hexz_pb2.AddTrainingExamplesRequest(model_key=model_key)
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
                if (
                    resp.status
                    != hexz_pb2.AddTrainingExamplesResponse.REJECTED_TRAINING
                ):
                    break
                # Server is training. Back off exponentially (with 10% jitter) and try again later.
                backoff_secs = (1.5**i) * np.random.uniform(0.90, 1.10)
                self.logger.info(
                    f"Server is training. Backing off for {backoff_secs:.2f}s (#{i})"
                )
                time.sleep(backoff_secs)
            if resp.status == hexz_pb2.AddTrainingExamplesResponse.REJECTED_WRONG_MODEL:
                # Load newer model
                model_key, model = self.fetch_model(resp.latest_model)
                self.logger.info(
                    f"Using new model {model_key.name}:{model_key.checkpoint}"
                )
            elif resp.status != hexz_pb2.AddTrainingExamplesResponse.ACCEPTED:
                raise HexzError(
                    f"Unexpected return code from training server: {hexz_pb2.AddTrainingExamplesResponse.Status.Name(resp.status)}"
                )
        tm.print_perf_stats()


def main():
    config = WorkerConfig.from_env()
    # OMG, what a freaking mess Python logging is...
    logging.config.dictConfig(
        {
            "version": 1,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "SelfPlayWorker": {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": False,
                }
            },
        }
    )
    worker = SelfPlayWorker(config)
    worker.logger.setLevel(logging.INFO)

    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"torch version: {torch.__version__}")
    if config.device == "cuda" and not torch.cuda.is_available():
        print("Device cuda not available, falling back to cpu.")
        config.device = "cpu"
    elif config.device == "mps" and not torch.backends.mps.is_available():
        print("Device mps not available, falling back to cpu.")
        config.device = "cpu"

    try:
        worker.generate_examples()
    except HexzError as e:
        worker.logger.error(f"generate_examples failed: {e}")
        raise


if __name__ == "__main__":
    main()
