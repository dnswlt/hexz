"""Implementation of a "model server", which can serve existing models and examples
from local storage as well as from GCS. It also accepts uploads of new model checkpoints
and example zip archives.
"""

from collections.abc import Iterable
import datetime
import gzip
import os
import pytz
import re
from typing import Optional
import torch

from pyhexz import hexz_pb2
from pyhexz.hexz import HexzNeuralNetwork


class ModelRepository:
    """Interface type for local and remote model repositories."""

    def get_latest_checkpoint(self, name: str) -> Optional[int]:
        """Returns the latest checkpoint number for the given model, or None if no checkpoint exists."""
        pass

    def get_model(
        self,
        name: str,
        checkpoint: int = None,
        as_bytes: bool = False,
        repr: str = "state_dict",
    ) -> HexzNeuralNetwork | bytes:
        """Loads the given model name and checkpoint from the repository.

        Args:
            as_bytes: If True, the raw serialized PyTorch model bytes are returned.
        """
        pass

    def store_model(
        self,
        name: str,
        checkpoint: int,
        model: HexzNeuralNetwork,
        overwrite=False,
        store_sm=True,
    ) -> str:
        """Stores the given model in the repository as a state_dict.

        Args:
            overwrite: if True, overwrites any existing model with the same name and checkpoint.
            store_sm: if True, stores a torch.jit.ScriptModule in addition to the state_dict.
        """
        pass

    def add_examples(self, req: hexz_pb2.AddTrainingExamplesRequest) -> None:
        """Saves the given request req in an examples/ subfolder of the model checkpoint."""
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

    def _scriptmodule_path(self, name: str, checkpoint: int):
        return os.path.join(
            self._model_base(name), "checkpoints", str(checkpoint), "scriptmodule.pt"
        )

    def get_latest_checkpoint(self, name: str) -> Optional[int]:
        cpdir = os.path.join(self._model_base(name), "checkpoints")
        regex = re.compile(r"^\d+$")
        try:
            return max(
                (int(d) for d in os.listdir(cpdir) if regex.match(d)), default=None
            )
        except FileNotFoundError:
            # Raised by os.listdir if cpdir does not exist.
            return None

    def get_model(
        self,
        name: str,
        checkpoint: int = None,
        map_location="cpu",
        as_bytes=False,
        repr="state_dict",
    ) -> HexzNeuralNetwork | torch.jit.ScriptModule | bytes:
        if checkpoint is None:
            checkpoint = self.get_latest_checkpoint(name)
        if repr == "scriptmodule":
            p = self._scriptmodule_path(name, checkpoint)
            if as_bytes:
                with open(p, "rb") as f_in:
                    return f_in.read()
            return torch.jit.load(p)
        # Treat as state_dict
        p = self._model_path(name, checkpoint)
        if as_bytes:
            with open(p, "rb") as f_in:
                return f_in.read()
        model = HexzNeuralNetwork()
        model.load_state_dict(torch.load(p, map_location=map_location))
        return model

    def store_model(
        self,
        name: str,
        checkpoint: int,
        model: HexzNeuralNetwork,
        overwrite=False,
        store_sm=True,
    ) -> str:
        m_path = self._model_path(name, checkpoint)
        if not overwrite and os.path.exists(m_path):
            raise IOError(f"Model already exists at {m_path}")
        os.makedirs(os.path.dirname(m_path), exist_ok=True)
        torch.save(model.state_dict(), m_path)
        if store_sm:
            # Generate and store a ScriptModule as well.
            sm_path = self._scriptmodule_path(name, checkpoint)
            sm = torch.jit.script(model)
            sm.save(sm_path)
        return m_path

    def add_examples(self, req: hexz_pb2.AddTrainingExamplesRequest) -> None:
        name = req.model_key.name
        checkpoint = req.model_key.checkpoint
        d = os.path.dirname(self._model_path(name, checkpoint))
        examples_dir = os.path.join(d, "examples")
        os.makedirs(examples_dir, exist_ok=True)
        now = datetime.datetime.now(tz=pytz.UTC).strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(examples_dir, f"{now}.gz")
        with gzip.open(filename, "wb") as f:
            f.write(req.SerializeToString())
