from contextlib import contextmanager
import datetime
import queue
from flask import Flask, current_app, make_response, request
from google.protobuf.message import DecodeError
import logging
import pytz

from pyhexz import hexz_pb2
from pyhexz.ccapi import CppMoveSuggester
from pyhexz.config import CPUEngineConfig


def create_app():
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)
    app.cpu_engine_config = CPUEngineConfig.from_env()
    app.move_suggester_queue = queue.SimpleQueue()

    @contextmanager
    def get_move_suggester():
        """Gets a CppMoveSuggester from the pool, if one is available. Otherwise, loads a new one.

        Each thread processing a request needs exclusive access to a CppMoveSuggester instance.
        """
        try:
            inst = current_app.move_suggester_queue.get_nowait()
        except queue.Empty:
            inst = CppMoveSuggester(current_app.cpu_engine_config.local_model_path)
        try:
            yield inst
        finally:
            current_app.move_suggester_queue.put(inst)
            
    # The path /hexz/cpu/suggest must be identical to the one the Go client uses.
    @app.post("/hexz/cpu/suggest")
    def suggestmove():
        """This request is used when using the server as a remote CPU player."""
        # ge_state = hexz_pb2.GameEngineState().FromString(base64.b64decode(enc_state))
        with get_move_suggester() as cms:
            try:
                return cms.suggest_move(request.data), {
                    "Content-Type": "application/x-protobuf"
                }
            except ValueError as e:
                # Cython converts std::invalid_argument to ValueError for us.
                return str(e), 400
            except RuntimeError as e:
                # Cython converts std::runtime_error to RuntimeError for us.
                return str(e), 500

    @app.get("/")
    def index():
        now = datetime.datetime.now(tz=pytz.UTC).isoformat()
        resp = make_response(f"Hello from Python hexz at {now}!\n")
        resp.headers["Content-Type"] = "text/plain"
        return resp

    return app
