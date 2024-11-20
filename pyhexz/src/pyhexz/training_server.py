from concurrent import futures
import datetime
import pprint
import grpc

import io
from flask import Flask, make_response
from google.protobuf import json_format
import logging
import pytz
import threading


from pyhexz.config import TrainingConfig
from pyhexz.errors import HexzError
from pyhexz.model import HexzNeuralNetwork
from pyhexz.training import TrainingState
from pyhexz import hexz_pb2
from pyhexz import hexz_pb2_grpc
from pyhexz import svg
from pyhexz.modelrepo import LocalModelRepository, ModelRepository
from pyhexz import training


class TrainingServicer(hexz_pb2_grpc.TrainingServiceServicer):

    def __init__(
        self,
        logger: logging.Logger,
        model_repo: LocalModelRepository,
        training_state: TrainingState,
        config: TrainingConfig,
    ):
        self.logger = logger
        self.model_repo = model_repo
        self.training_state = training_state
        self.config = config

    def AddTrainingExamples(
        self, request: hexz_pb2.AddTrainingExamplesRequest, ctx: grpc.ServicerContext
    ):
        self.logger.info(
            f"Received AddTrainingExamplesRequest with {len(request.examples)} examples from {request.execution_id}."
        )
        # sanity checks on the request to ignore data from rogue workers.
        if request.worker_config.runs_per_move < self.config.min_runs_per_move:
            self.logger.warning(
                f"Ignoring AddTrainingExamplesRequest from {request.execution_id}: runs_per_move too low. "
                f"Want >={self.config.min_runs_per_move}, got {request.worker_config.runs_per_move}"
            )
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "runs_per_move too low")
        if not self.training_state.accept(request):
            self.logger.warning(
                f"Ignoring AddTrainingExamplesRequest from {request.execution_id}: runs_per_move too low. "
                f"Want >={self.config.min_runs_per_move}, got {request.worker_config.runs_per_move}"
            )
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid model_key")

        self.training_state.add_examples(request)
        return hexz_pb2.AddTrainingExamplesResponse(
            status=hexz_pb2.AddTrainingExamplesResponse.ACCEPTED,
            latest_model=self.training_state.model_key(),
        )

    def FetchModel(
        self, request: hexz_pb2.FetchModelRequest, ctx: grpc.ServicerContext
    ):
        if request.HasField("model_key"):
            self.logger.info(
                f"Received FetchModelRequest for model_key {json_format.MessageToJson(request.model_key, indent=None)}"
            )
            model_key = request.model_key
            if model_key.checkpoint == -1:
                model_key.checkpoint = self.training_state.latest_checkpoint
        else:
            model_key = self.training_state.model_key()
            self.logger.info(
                f"Received FetchModelRequest for latest model ({json_format.MessageToJson(model_key, indent=None)})"
            )
        repr = "scriptmodule"
        if request.encoding == hexz_pb2.ModelEncoding.STATE_DICT:
            repr = "state_dict"
        elif request.encoding == hexz_pb2.ModelEncoding.JIT_TRACE:
            ctx.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Encoding JIT_TRACE not supported yet.",
            )

        model = self.model_repo.get_model(
            model_key.name, model_key.checkpoint, as_bytes=True, repr=repr
        )
        return hexz_pb2.FetchModelResponse(
            model_key=model_key,
            encoding=request.encoding,
            model_bytes=model,
        )

    def ControlEvents(
        self, request: hexz_pb2.ControlRequest, ctx: grpc.ServicerContext
    ):
        self.logger.info(
            f"Worker {ctx.peer()} with execution_id {request.execution_id} subscribed to control events."
        )
        yield from self.training_state.subscribe_events(ctx)
        self.logger.info(
            f"Worker with execution_id {request.execution_id} disconnected."
        )


_grpc_server = None


def handle_shutdown(signum, frame):
    print(f"Received signal %d. Stopping gRPC server.", signum)
    _grpc_server.stop(None)


def serve_grpc(
    logger: logging.Logger,
    model_repo: LocalModelRepository,
    training_state: TrainingState,
    config: TrainingConfig,
):
    global _grpc_server

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    svc = TrainingServicer(logger, model_repo, training_state, config)
    hexz_pb2_grpc.add_TrainingServiceServicer_to_server(svc, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    _grpc_server = server
    server.wait_for_termination()


def init_repo_if_missing(app: Flask) -> None:
    repo: ModelRepository = app.model_repo
    config: TrainingConfig = app.hexz_config
    name = config.model_name
    if repo.get_latest_checkpoint(name) is not None:
        return
    model_type = config.model_type
    blocks = config.model_blocks
    model = HexzNeuralNetwork(model_type=model_type, blocks=blocks)
    repo.store_model(name, 0, model)
    app.logger.info(f"Created new initial model in repo for model '{name}'")


def create_app():
    # Use the Flask app only for HTML status pages.
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
    training_state = training.TrainingState(
        model_repo, config.model_name, logger=app.logger, config=config
    )
    app.training_state = training_state

    # Start gRPC server in separate thread.
    grpc_thread = threading.Thread(
        target=serve_grpc, args=(app.logger, model_repo, training_state, config)
    )
    grpc_thread.start()
    # signal.signal(signal.SIGINT, handle_shutdown)
    # signal.signal(signal.SIGTERM, handle_shutdown)
    app.grpc_thread = grpc_thread

    @app.get("/")
    def index():
        now = datetime.datetime.now(tz=pytz.UTC).isoformat()
        ts: TrainingState = app.training_state
        stats = ts.stats()
        lines = [
            f"Hello from Python hexz at {now}!",
            f"{pprint.pformat(app.hexz_config)}",
            f"{pprint.pformat(stats)}",
        ]
        resp = make_response("\n\n".join(lines))
        resp.headers["Content-Type"] = "text/plain"
        return resp

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

    return app
