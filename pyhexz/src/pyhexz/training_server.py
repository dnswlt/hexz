from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import dataclasses
import datetime
import io
import time
from typing import Any
from flask import Flask, make_response, request
from google.protobuf.message import DecodeError
from google.protobuf import json_format
import logging
import pytz
import threading

from pyhexz.hexc import CBoard
from pyhexz.config import TrainingConfig
from pyhexz.errors import HexzError
from pyhexz.hexz import HexzNeuralNetwork
from pyhexz import hexz_pb2
from pyhexz import svg
from pyhexz.modelrepo import LocalModelRepository, ModelRepository
from pyhexz import training


@dataclass
class TrainingStats:
    example_requests: int = 0
    examples: int = 0
    training_runs: int = 0


class TrainingTask:
    """TrainingTask is responsible for training the model given the stored examples.

    Each task should be executed by an individual thread or in a thread pool.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint: int,
        logger: logging.Logger,
    ):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.logger = logger

    def execute(self):
        self.logger.info(
            "Someone tried to run training. That's nice, but we don't have that implemented yet. I'll sleep for a bit, pretending to do work."
        )
        t_start = time.time()
        time.sleep(5)
        self.logger.info(f"Training (fake) done after {time.time()-t_start:.3f}s")


class TrainingState:
    """TrainingState is the entry point for adding examples and training the model.

    A training server should maintain a single instance of TrainingState.
    Instances are thread-safe. All methods can be called from multiple threads.

    Attributes:
        training_examples (int): Number of examples to collect before starting a
            new training run.
    """

    train_after_examples: int = 100

    def __init__(
        self, model_repo: ModelRepository, model_name: str, logger: logging.Logger
    ):
        self.lock = threading.Lock()
        self.model_repo = model_repo
        self.model_name = model_name
        self.logger = logger
        self.latest_checkpoint = model_repo.get_latest_checkpoint(model_name)
        self._stats = TrainingStats()
        self._latest_request: hexz_pb2.AddTrainingExamplesRequest = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.last_training = 0  # Example count when the last training was started.
        self.is_training: bool = False

    def model_key(self):
        """Returns the latest model key."""
        with self.lock:
            return hexz_pb2.ModelKey(
                name=self.model_name, checkpoint=self.latest_checkpoint
            )

    def _should_train(self) -> bool:
        """Must only be called while holding the lock."""
        return (
            self._stats.examples >= self.last_training + self.train_after_examples
            and not self.is_training
        )

    def _start_training(self):
        """Must only be called while holding the lock."""
        task = TrainingTask(self.model_name, self.latest_checkpoint, logger=self.logger)
        self.logger.info(f"Starting a new TrainingTask at {self._stats.examples} examples")
        fut = self._executor.submit(task.execute)
        fut.add_done_callback(self.training_done)
        self.is_training = True
        self.last_training = self._stats.examples
        self._stats.training_runs += 1

    def add_examples(self, req: hexz_pb2.AddTrainingExamplesRequest):
        """Saves the examples from the given request in the repository.

        If enough new examples have been collected, a new training is started,
        unless one is already ongoing.
        """
        self.model_repo.add_examples(req)
        with self.lock:
            self._latest_request = req
            self._stats.example_requests += 1
            self._stats.examples += len(req.examples)
            if self._should_train():
                self._start_training()

    def stats(self):
        """Returns a copy of the training stats."""
        with self.lock:
            return dataclasses.replace(self._stats)

    def latest_request(self) -> hexz_pb2.AddTrainingExamplesRequest:
        with self.lock:
            return self._latest_request

    def training_done(self, fut: Future[Any]):
        """A callback that is called when a TrainingTask is done."""
        self.logger.info("TrainingState.training_done")
        with self.lock:
            self.is_training = False
            # Trigger next training round immediately if there are enough
            # examples already.
            if self._should_train():
                self._start_training()


def init_repo_if_missing(app: Flask) -> None:
    repo: ModelRepository = app.model_repo
    name: str = app.hexz_config.model_name
    if repo.get_latest_checkpoint(name) is not None:
        return
    model = HexzNeuralNetwork()
    repo.store_model(name, 0, model)
    app.logger.info(f"Created new initial model in repo for model '{name}'")


def create_app():
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)
    config = TrainingConfig.from_env()
    app.logger.info(f"Using {config}")
    app.hexz_config = config

    if not config.model_repo_base_dir:
        raise HexzError("No model_repo_base_dir specified.")
    if not config.model_name:
        raise HexzError(
            "No model_name specified. Did you forget to set HEXZ_MODEL_NAME?"
        )

    model_repo = LocalModelRepository(config.model_repo_base_dir)
    app.model_repo = model_repo
    init_repo_if_missing(app)
    app.training_state = TrainingState(model_repo, config.model_name, logger=app.logger)

    @app.get("/")
    def index():
        now = datetime.datetime.now(tz=pytz.UTC).isoformat()
        stats = app.training_state.stats()
        lines = [
            f"Hello from Python hexz at {now}!",
            f"Config: {app.hexz_config}",
            f"Training stats: {stats}",
        ]
        resp = make_response("\n\n".join(lines))
        resp.headers["Content-Type"] = "text/plain"
        return resp

    @app.post("/examples")
    def examples():
        """Part of the training workflow. Called by workers to upload new examples."""
        try:
            app.logger.info(
                f"Received AddTrainingExamplesRequest: size={len(request.data)}"
            )
            req = hexz_pb2.AddTrainingExamplesRequest.FromString(request.data)
        except DecodeError as e:
            return "Invalid AddTrainingExamplesRequest protocol buffer", 400
        if not req.examples:
            return "No examples in request", 400
        ts: TrainingState = app.training_state
        ts.add_examples(req)
        resp = hexz_pb2.AddTrainingExamplesResponse(
            status=hexz_pb2.AddTrainingExamplesResponse.ACCEPTED,
            latest_model=ts.model_key(),
        )
        return resp.SerializeToString(), {"Content-Type": "application/x-protobuf"}

    @app.get("/examples/latest")
    def examples_html():
        """Returns a HTML file with SVG images of the latest example batch."""
        ts: TrainingState = app.training_state
        req: hexz_pb2.AddTrainingExamplesRequest = ts.latest_request()
        if not req:
            return "No examples yet", 404
        npexs = [training.NumpyExample.decode(e) for e in req.examples]
        buf = io.StringIO()
        svg.export(
            buf,
            boards=[CBoard.from_numpy(e.board) for e in npexs],
            captions=[
                f"Move: {x.move.move} &bull; Value: {e.value[0]:.3f} PredVal: {x.model_predictions.value:.3f}"
                for x, e in zip(req.examples, npexs)
            ],
            move_probs=[e.move_probs for e in npexs],
            header=f"Execution ID: {req.execution_id}",
        )
        return buf.getvalue(), {"Content-Type": "text/html; charset=utf-8"}

    @app.get("/models/latest")
    def latest_model():
        """Part of the training workflow. Called by workers to download the latest
        model straight away. The model key is sent back JSON-encoded in the X-Model-Key header.
        (The idea being that this allows us to e.g. curl GET a model without any JSON/Proto fiddling.)
        """
        repr = request.args.get("repr", "state_dict").lower()
        if repr not in ("state_dict", "scriptmodule"):
            return "repr must be state_dict or scriptmodule", 400
        ts: TrainingState = app.training_state
        repo: ModelRepository = app.model_repo
        model_key = ts.model_key()
        try:
            mbytes = repo.get_model(
                model_key.name, model_key.checkpoint, as_bytes=True, repr=repr
            )
        except FileNotFoundError as e:
            app.logger.error(f"Cannot load model {model_key}: {e}")
            "no latest model", 500
        return mbytes, {
            "Content-Type": "application/octet-stream",
            "X-Model-Key": json_format.MessageToJson(model_key, indent=None),
        }

    @app.get("/models/<name>/checkpoints/<int:checkpoint>")
    def model_bytes(name, checkpoint):
        """Returns the requested model in raw bytes.
        Will only return data if the requested model is currently relevant for training.
        (The reason is that we don't want to turn our server into a model download server just yet.)

        This returns the raw bytes of the PyTorch encoded model and not a protobuf or JSON.
        """
        return "not implemented", 404

    # For debugging bad requests:
    # @app.errorhandler(400)
    # def handle_bad_request(e):
    #     print("Really bad request!!!", e)
    #     return 'bad request!', 400

    return app
