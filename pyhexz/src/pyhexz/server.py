import base64
from collections.abc import Iterable
from contextlib import contextmanager
import datetime
import io
from flask import Flask, current_app, make_response, request
from google.protobuf.message import DecodeError
from google.protobuf import json_format
import json
import logging
import pytz
import queue
import time
import typing

from pyhexz.hexc import CBoard
from pyhexz.config import TrainingConfig
from pyhexz.errors import HexzError
from pyhexz.hexz import HexzNeuralNetwork, NeuralMCTS
from pyhexz import hexz_pb2
from pyhexz import sconv
from pyhexz import svg
from pyhexz.modelrepo import LocalModelRepository, ModelRepository
from pyhexz import training


def suggest_move(
    model: HexzNeuralNetwork, state: hexz_pb2.GameEngineState, think_time: float
) -> tuple[dict[str, typing.Any], int]:
    """Runs the ML model to obtain a move suggestion.

    Returns:
        A tuple of (SuggestMoveResponse JSON response, http status code).

    Args:
        model: the model to use in neural MCTS.
        state: the game state, including the player whose turn it is, and the board.
        think_time: thinking time in seconds.
    """
    board = CBoard.from_numpy(sconv.convert_board(state.flagz.board))
    try:
        board.validate()
    except ValueError as e:
        return str(e), 400
    turn = state.flagz.board.turn - 1  # Go impl uses (1, 2), we use (0, 1).
    print(f"Board info: flags:({board.flags(0)},{board.flags(0)}), turn:{turn}")
    mcts = NeuralMCTS(
        board, model, game_id=time.strftime("CPU-%Y%m%d-%H%M%S"), turn=turn
    )
    n = 0
    started = time.perf_counter()
    while True:
        if n & 63 == 0 and time.perf_counter() - started >= think_time:
            break
        mcts.run()
        n += 1
    best_child = mcts.root.best_child()
    if not best_child:
        raise ValueError("No next move")
    typ, r, c, _ = best_child._move
    # Return a SuggestMoveResponse JSON.
    resp = {
        "move": {
            "move": state.flagz.board.move,
            "row": int(r),
            "col": int(c),
            "type": 0 if typ == 1 else 5,  # 5==Flag
        }
    }
    svg.export(
        "./latest.html",
        [board, board],
        [f"Model move probs (value: {mcts.value:.3f})", "MCTS move likelihoods"],
        [mcts.root._move_probs, mcts.root.move_likelihoods()],
    )
    print(f"child qs: {[(c._wins, c._visit_count) for c in mcts.root._children]}")
    print(
        f"suggested move: {resp}, iterations: {n}, tree size: {mcts.size()}, best child: vc:{best_child._visit_count} {best_child._wins} {best_child.puct()}."
    )
    return resp, 200


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
    app.model_queue = queue.SimpleQueue()
    app.hexz_config = config

    if config.model_repo_base_dir:
        app.training_task_queue = queue.SimpleQueue()
        app.model_repo = LocalModelRepository(config.model_repo_base_dir)
        if not config.model_name:
            raise HexzError(
                "No model_name specified. Did you forget to set HEXZ_MODEL_NAME?"
            )
        init_repo_if_missing(app)
        # Start the background training task.
        t = training.TrainingTask(
            repo=app.model_repo,
            config=config,
            queue=app.training_task_queue,
            logger=app.logger,
        )
        t.daemon = True
        t.start()
        app.training_task = t

    @contextmanager
    def get_model():
        """Gets a model from the model_queue, if one is available. Otherwise, loads a new one.

        The idea is that each thread processing a request needs exclusive access to a model.
        """
        try:
            model = current_app.model_queue.get_nowait()
        except queue.Empty:
            model = app.model_repo.get_model(app.hexz_config.model_name)
        try:
            yield model
        finally:
            current_app.model_queue.put(model)

    # The path /hexz/cpu/suggest must be identical to the one the Go client uses.
    @app.post("/hexz/cpu/suggest")
    def suggestmove():
        """This request is used when using the server as a remote CPU player."""
        req = request.json
        think_time_ns = req["maxThinkTime"]
        enc_state = req["gameEngineState"]
        ge_state = hexz_pb2.GameEngineState().FromString(base64.b64decode(enc_state))
        if not current_app.model_path:
            return "Missing model_path", 500
        with get_model() as model:
            return suggest_move(model, ge_state, think_time_ns / 1e9)

    @app.get("/")
    def index():
        now = datetime.datetime.now(tz=pytz.UTC).isoformat()
        resp = make_response(f"Hello from Python hexz at {now}!\n")
        resp.headers["Content-Type"] = "text/plain"
        return resp

    @app.get("/status")
    def status():
        now = datetime.datetime.now(tz=pytz.UTC).isoformat(timespec="milliseconds")
        reply_q = queue.SimpleQueue()
        current_app.training_task_queue.put(
            {
                "type": "GetTrainingInfo",
                "reply_q": reply_q,
            }
        )
        training_info = reply_q.get(timeout=5)
        return json.dumps(
            {
                "timestamp": now,
                "training": training_info,
            },
            indent=True,
        ), {"Content-Type": "application/json"}

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
        reply_q = queue.SimpleQueue()
        current_app.training_task_queue.put(
            {
                "type": "AddTrainingExamplesRequest",
                "request": req,
                "reply_q": reply_q,
            }
        )
        resp: hexz_pb2.AddTrainingExamplesResponse = reply_q.get(timeout=5)
        return resp.SerializeToString(), {"Content-Type": "application/x-protobuf"}

    @app.get("/examples/latest")
    def examples_html():
        """Returns a HTML file with SVG images of the latest example batch."""
        reply_q = queue.SimpleQueue()
        current_app.training_task_queue.put(
            {
                "type": "GetLatestExamples",
                "reply_q": reply_q,
            }
        )
        examples: Iterable[hexz_pb2.TrainingExample] = reply_q.get(timeout=5)
        npexs = [training.NumpyExample.decode(e) for e in examples]
        buf = io.StringIO()
        svg.export(buf, [CBoard.from_numpy(e.board) for e in npexs], [], [e.move_probs for e in npexs])
        return buf.getvalue(), {"Content-Type": "text/html; charset=utf-8"}

    @app.get("/models/current")
    def model():
        """Part of the training workflow. Called by workers to retrieve information
        about the model to use when generating examples.

        This returns JSON and not a protobuf so that we can also get this information
        from a web browser.
        """
        reply_q = queue.SimpleQueue()
        current_app.training_task_queue.put(
            {
                "type": "GetModelKey",
                "reply_q": reply_q,
            }
        )
        model_key: hexz_pb2.ModelKey = reply_q.get(timeout=5)
        return json_format.MessageToJson(model_key), {
            "Content-Type": "application/json",
        }

    @app.get("/models/latest")
    def latest_model():
        """Part of the training workflow. Called by workers to download the latest
        model straight away. The model key is sent JSON-encoded in the X-Model-Key header.
        """
        repr = request.args.get("repr", "state_dict").lower()
        if repr not in ("state_dict", "scriptmodule"):
            return "repr must be state_dict or scriptmodule", 400
        reply_q = queue.SimpleQueue()
        current_app.training_task_queue.put(
            {
                "type": "GetModel",
                "repr": repr,
                "reply_q": reply_q,
            }
        )
        r = reply_q.get(timeout=5)
        if r.get("error") is not None:
            return r["error"], 500
        return r["data"], {
            "Content-Type": "application/octet-stream",
            "X-Model-Key": json_format.MessageToJson(r["model_key"], indent=None),
        }

    @app.get("/models/<name>/checkpoints/<int:checkpoint>")
    def model_bytes(name, checkpoint):
        """Returns the requested model in raw bytes.
        Will only return data if the requested model is currently relevant for training.
        (The reason is that we don't want to turn our server into a model download server just yet.)

        This returns the raw bytes of the PyTorch encoded model and not a protobuf or JSON.
        """
        repr = request.args.get("repr", "state_dict").lower()
        if repr not in ("state_dict", "scriptmodule"):
            return "repr must be state_dict or scriptmodule", 400
        reply_q = queue.SimpleQueue()
        current_app.training_task_queue.put(
            {
                "type": "GetModel",
                "model_key": hexz_pb2.ModelKey(
                    name=name,
                    checkpoint=checkpoint,
                ),
                "repr": repr,
                "reply_q": reply_q,
            }
        )
        r = reply_q.get(timeout=5)
        if r.get("error") is not None:
            return r["error"], 404
        return r["data"], {
            "Content-Type": "application/octet-stream",
            "X-Model-Key": json_format.MessageToJson(r["model_key"], indent=None),
        }

    # For debugging bad requests:
    # @app.errorhandler(400)
    # def handle_bad_request(e):
    #     print("Really bad request!!!", e)
    #     return 'bad request!', 400

    return app
