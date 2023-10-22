"""Implementation of a "model server", which can serve existing models and examples
from local storage as well as from GCS. It also accepts uploads of new model checkpoints
and example zip archives.
"""

import datetime
from flask import Flask, current_app, redirect, request, url_for
import io
import logging
import os
import pytz
import re
from typing import Optional
from werkzeug.utils import secure_filename

from pyhexz.errors import HexzError

class ModelRepository:
    """Interface type for local and remote model repositories."""
    def get_latest_checkpoint(self, name: str) -> Optional[int]:
        pass
    def get_model(self, name: str, checkpoint: int) -> bytes:
        pass
    def store_model(self, name: str, checkpoint: int, data: bytes, overwrite=False) -> str:
        pass

class LocalModelRepository:
    """A model repository using local file system storage."""
    def __init__(self, basedir):
        self.basedir = basedir
    
    def _model_base(self, name: str):
        return os.path.join(self.basedir, "models", "flagz", name)
    
    def _model_path(self, name: str, checkpoint: int):
        return os.path.join(self._model_base(name), "checkpoints", str(checkpoint), "model.pt")

    def get_latest_checkpoint(self, name: str) -> Optional[int]:
        """Returns the latest checkpoint number for the given model, or None if no checkpoint exists."""
        cpdir = os.path.join(self._model_base(name), "checkpoints")
        regex = re.compile(r'^\d+$')
        try:
            return max((int(d) for d in os.listdir(cpdir) if regex.match(d)), default=None)
        except FileNotFoundError:
            # Raised by os.listdir if cpdir does not exist.
            return None

    def get_model(self, name: str, checkpoint: int) -> bytes:
        """Loads the given model name and checkpoint from the repository.
        
        The returned bytes typically represent a PyTorch saved model.
        """
        p = self._model_path(name, checkpoint)
        with open(p, "rb") as f_in:
            return f_in.read()
    
    def store_model(self, name: str, checkpoint: int, data: bytes, overwrite=False) -> str:
        p = self._model_path(name, checkpoint)
        if not overwrite and os.path.exists(p):
            raise IOError(f"Model already exists at {p}")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f_out:
            f_out.write(data)
        return p


def expect_env(name):
    v = os.getenv(name)
    if not v:
        raise HexzError(f"environment variable {name} is not set.")
    return v


def create_app():
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)

    if os.getenv("GCS_BUCKET"):
        # TODO: implement GCS model repo.
        raise HexzError("Not yet implemented")
    else:
        app.model_repo = LocalModelRepository(expect_env("MODEL_BASE_DIR"))

    @app.route("/models/<name>/checkpoints/<int:checkpoint>", methods=["GET"])
    def model_checkpoint(name: str, checkpoint: int):
        try:
            model = current_app.model_repo.get_model(name, checkpoint)
            return model, {"Content-Type": "application/octet-stream"}
        except FileNotFoundError as e:
            return "", 404

    @app.route("/models/<name>", methods=["POST", "GET"])
    def models(name: str):
        repo: ModelRepository = current_app.model_repo
        latest = repo.get_latest_checkpoint(name)
        if request.method == "GET":
            if latest is not None:
                model = repo.get_model(name, latest)
                return model, {"Content-Type": "application/octet-stream"}
            else:
                return "", 404
        # Append new model.
        next_cp = latest + 1 if latest is not None else 0
        if "model" not in request.files:
            return "No model uploaded", 400
        file = request.files["model"]
        data = io.BytesIO()
        file.save(data)
        model_path = repo.store_model(name, next_cp, data.getvalue())
        current_app.logger.info(f"Stored new model at {model_path}")
        return redirect(url_for('model_checkpoint', name=name, checkpoint=next_cp))

    @app.get("/status")
    def status():
        now = datetime.datetime.now(tz=pytz.UTC).isoformat(timespec='milliseconds')
        return {
            "timestamp": now,
        }

    return app
