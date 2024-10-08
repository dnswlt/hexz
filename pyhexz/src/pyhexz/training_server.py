import datetime
import io
import typing
from flask import Flask, make_response, request
from google.protobuf.message import DecodeError
from google.protobuf import json_format
import logging
import numpy as np
import pytz
import torch

from pyhexz.board import Board
from pyhexz.config import TrainingConfig
from pyhexz.errors import HexzError
from pyhexz.model import HexzNeuralNetwork
from pyhexz.training import TrainingState
from pyhexz import hexz_pb2
from pyhexz import svg
from pyhexz.modelrepo import LocalModelRepository, ModelRepository
from pyhexz import training


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
    app.training_state = training.TrainingState(
        model_repo, config.model_name, logger=app.logger, config=config
    )

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
        buf = io.StringIO()
        svg.export(buf, req)
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

    # For debugging bad requests:
    # @app.errorhandler(400)
    # def handle_bad_request(e):
    #     print("Really bad request!!!", e)
    #     return 'bad request!', 400

    return app
