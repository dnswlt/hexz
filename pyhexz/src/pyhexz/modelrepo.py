"""Implementation of a "model server", which can serve existing models and examples
from local storage as well as from GCS. It also accepts uploads of new model checkpoints
and example zip archives.
"""

from collections.abc import Iterable
import datetime
import os
import pytz
import re
from typing import Optional
import torch
import zipfile

from pyhexz import hexz_pb2
from pyhexz.hexz import HexzNeuralNetwork


class ModelRepository:
    """Interface type for local and remote model repositories."""

    def get_latest_checkpoint(self, name: str) -> Optional[int]:
        pass

    def get_model(self, name: str, checkpoint: int = None, as_bytes: bool = False) -> (HexzNeuralNetwork | bytes):
        pass

    def store_model(
        self, name: str, checkpoint: int, model: HexzNeuralNetwork, overwrite=False
    ) -> str:
        pass

    def add_examples(
        self, name: str, checkpoint: int, examples: Iterable[hexz_pb2.TrainingExample]
    ):
        pass


class LocalModelRepository:
    """A model repository using local file system storage."""

    def __init__(self, basedir: str):
        self.basedir = basedir

    def _model_base(self, name: str):
        return os.path.join(self.basedir, "models", "flagz", name)

    def _model_path(self, name: str, checkpoint: int):
        return os.path.join(
            self._model_base(name), "checkpoints", str(checkpoint), "model.pt"
        )

    def get_latest_checkpoint(self, name: str) -> Optional[int]:
        """Returns the latest checkpoint number for the given model, or None if no checkpoint exists."""
        cpdir = os.path.join(self._model_base(name), "checkpoints")
        regex = re.compile(r"^\d+$")
        try:
            return max(
                (int(d) for d in os.listdir(cpdir) if regex.match(d)), default=None
            )
        except FileNotFoundError:
            # Raised by os.listdir if cpdir does not exist.
            return None

    def get_model(self, name: str, checkpoint: int = None, map_location="cpu", as_bytes=False) -> (HexzNeuralNetwork | bytes):
        """Loads the given model name and checkpoint from the repository.

        The returned bytes typically represent a PyTorch saved model.
        """
        if checkpoint is None:
            checkpoint = self.get_latest_checkpoint(name)
        p = self._model_path(name, checkpoint)
        if as_bytes:
            with open(p, "rb") as f_in:
                return f_in.read()
        model = HexzNeuralNetwork()
        model.load_state_dict(torch.load(p, map_location=map_location))
        return model

    def store_model(
        self, name: str, checkpoint: int, model: HexzNeuralNetwork, overwrite=False
    ) -> str:
        p = self._model_path(name, checkpoint)
        if not overwrite and os.path.exists(p):
            raise IOError(f"Model already exists at {p}")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save(model.state_dict(), p)
        return p

    def add_examples(
        self, name: str, checkpoint: int, examples: Iterable[hexz_pb2.TrainingExample]
    ) -> None:
        d = os.path.dirname(self._model_path(name, checkpoint))
        os.makedirs(os.path.join(d, "examples"), exist_ok=True)
        now = datetime.datetime.now(tz=pytz.UTC).strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(d, f"{now}.zip")
        with zipfile.ZipFile(filename, "w") as zip:
            for i, ex in enumerate(examples):
                zip.writestr(f"{i:06d}.pb", ex.SerializeToString())
        