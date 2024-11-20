"""Implementation of a "model server", which can serve existing models and examples
from local storage as well as from GCS. It also accepts uploads of new model checkpoints
and example zip archives.
"""

from collections import deque
from contextlib import contextmanager
import datetime
import gzip
import io
import itertools
import json
import queue
import time
import typing
import h5py
import os
import threading
import numpy as np
import pytz
import re
from typing import Any, Iterator, Optional
import torch

from pyhexz import hexz_pb2
from pyhexz.model import HexzNeuralNetwork


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
            repr: Determines the model returned representation.
                One of "state_dict" or "scriptmodule".
        """
        pass

    def store_model(
        self,
        name: str,
        checkpoint: int | None,
        model: HexzNeuralNetwork,
        store_sm=True,
    ) -> str:
        """Stores the given model in the repository as a state_dict.

        Args:
            store_sm: if True, stores a torch.jit.ScriptModule in addition to the state_dict.
        """
        pass

    def add_examples(self, req: hexz_pb2.AddTrainingExamplesRequest) -> None:
        """Saves the given request req in an examples/ subfolder of the model checkpoint."""
        pass


class RequestIndexEntry(typing.NamedTuple):
    execution_id: str
    game_id: str
    examples: int
    # Path relative to repo's model base.
    relpath: str


class LocalModelRepository:
    """A model repository using local file system storage."""

    def __init__(self, basedir: str):
        self.basedir = basedir
        self.examples_lock = threading.Lock()
        self.models_lock = threading.Lock()
        # Stores the latest model, keyed by /{name}/{checkpoint}/bytes={as_bytes?}/repr={repr}
        self.model_cache: dict[str, Any] = {}
        # Map of open HDF5 files, keyed by file name.
        self.h5_files: dict[str, queue.SimpleQueue] = {}
        self.h5_lock = threading.Lock()

    def _model_base(self, name: str):
        return os.path.join(self.basedir, "models", "flagz", name)

    def _checkpoints_dir(self, name: str, checkpoint: int):
        return os.path.join(self._model_base(name), "checkpoints", str(checkpoint))

    def _model_path(self, name: str, checkpoint: int):
        return os.path.join(self._checkpoints_dir(name, checkpoint), "model.pt")

    def _scriptmodule_path(self, name: str, checkpoint: int):
        return os.path.join(self._checkpoints_dir(name, checkpoint), "scriptmodule.pt")

    def _examples_dir(self, name: str, checkpoint: int):
        return os.path.join(self._checkpoints_dir(name, checkpoint), "examples")

    def _request_index_path(self, name: str):
        """Returns the path to the jsonlines index file for example requests."""
        return os.path.join(self._model_base(name), "index", "requests.jsonl")

    def get_latest_checkpoint(self, name: str) -> Optional[int]:
        cpdir = os.path.join(self._model_base(name), "checkpoints")
        regex = re.compile(r"^\d+$")
        with self.models_lock:
            try:
                return max(
                    (int(d) for d in os.listdir(cpdir) if regex.match(d)), default=None
                )
            except FileNotFoundError:
                # Raised by os.listdir if cpdir does not exist.
                return None

    @contextmanager
    def acquire_h5(self, model_name):
        """Acquires and returns an exlusive handle to the HDF5 file for the specified model.

        The returned h5py.File is open in append ("a") mode and ready for reading and writing.
        The call blocks until the file is available.
        """
        h5_file = os.path.join(self._model_base(model_name), "h5", "examples.h5")
        with self.h5_lock:
            q = self.h5_files.get(h5_file)
            if q is None:
                if not os.path.exists(h5_file):
                    os.makedirs(os.path.dirname(h5_file), exist_ok=True)
                h = h5py.File(h5_file, "a")
                q = queue.SimpleQueue()
                q.put(h)
                self.h5_files[h5_file] = q
        try:
            h = q.get()
            yield h
        finally:
            q.put(h)
        return

    def h5_size(self, model_name):
        """Returns the number of examples stored in HDF5 for the given model."""
        h5_file = os.path.join(self._model_base(model_name), "h5", "examples.h5")
        if not os.path.exists(h5_file):
            return 0
        with self.acquire_h5(model_name) as h:
            return len(h["boards"])

    def close_all(self):
        """Closes all open file / HDF5 handles.

        The repository will remain usable and will re-open any relevant files as necessary.
        """
        with self.h5_lock:
            # We hold the lock, so no other threads can open new files. Now wait for
            # all handles to become available again (they are never used while also holding the h5_lock).
            for q in self.h5_files.values():
                h = q.get()
                h.close()
            self.h5_files.clear()

    def get_model(
        self,
        name: str,
        checkpoint: int | None = None,
        map_location="cpu",
        as_bytes=False,
        repr="state_dict",
    ) -> HexzNeuralNetwork | torch.jit.ScriptModule | bytes:
        if checkpoint in (-1, None):
            checkpoint = self.get_latest_checkpoint(name)

        key = f"/{name}/{checkpoint}/bytes={as_bytes}/repr={repr}"
        with self.models_lock:
            res = self.model_cache.get(key)
            if res:
                return res

        def _get_model(cp):
            if repr == "scriptmodule":
                p = self._scriptmodule_path(name, cp)
                if as_bytes:
                    with open(p, "rb") as f_in:
                        return f_in.read()
                return torch.jit.load(p)
            # Treat as state_dict
            p = self._model_path(name, cp)
            if as_bytes:
                with open(p, "rb") as f_in:
                    return f_in.read()
            with open(p + ".params", "r") as f_in:
                params = json.load(f_in)
            model = HexzNeuralNetwork(**params)
            model.load_state_dict(
                torch.load(p, map_location=map_location, weights_only=True)
            )
            return model

        model = _get_model(checkpoint)
        with self.models_lock:
            # For now, only cache the last requested model.
            self.model_cache.clear()
            self.model_cache[key] = model
        return model

    def store_model(
        self,
        name: str,
        checkpoint: int | None,
        model: HexzNeuralNetwork,
        store_sm=True,
    ) -> str:
        if checkpoint is None:
            checkpoint = self.get_latest_checkpoint(name) + 1
        m_path = self._model_path(name, checkpoint)
        with self.models_lock:
            if os.path.exists(m_path):
                raise IOError(f"Model already exists at {m_path}")
            os.makedirs(os.path.dirname(m_path), exist_ok=True)
            with open(m_path + ".params", "w") as f_out:
                json.dump(model.ctor_args, f_out, indent=2)
            torch.save(model.state_dict(), m_path)
            if store_sm:
                # Generate and store a ScriptModule as well.
                sm_path = self._scriptmodule_path(name, checkpoint)
                sm = torch.jit.script(model)
                sm.save(sm_path)
        return m_path

    def _index_add(
        self, model_name: str, filename: str, req: hexz_pb2.AddTrainingExamplesRequest
    ) -> None:
        """Adds the given request to the index, for fast lookup.

        Must only be called in a concurrent setup if examples_lock is held.
        """
        req_index = self._request_index_path(model_name)
        if not os.path.exists(req_index):
            os.makedirs(os.path.dirname(req_index))
        with open(req_index, "a", encoding="utf-8") as f_out:
            entry = {
                "execution_id": req.execution_id,
                "game_id": req.game_id,
                "examples": len(req.examples),
                "relpath": os.path.relpath(filename, self._model_base(model_name)),
            }
            json.dump(entry, f_out, indent=None)
            f_out.write("\n")

    def _index_iter(
        self, model_name: str, game_id: str, execution_id: str | None = None
    ) -> Iterator[RequestIndexEntry]:
        """Finds matching entries in the index and returns their relative paths.

        Callers must hold examples_lock when calling this method concurrently.
        """
        req_index = self._request_index_path(model_name)
        if not os.path.exists(req_index):
            return
        with open(req_index, "r", encoding="utf-8") as f_in:
            for lineno, line in enumerate(f_in, start=1):
                if not line.strip():
                    continue  # skip empty lines
                try:
                    entry = json.loads(line)
                    if execution_id and entry["execution_id"] != execution_id:
                        continue
                    if entry["game_id"] == game_id:
                        yield RequestIndexEntry(**entry)
                except json.JSONDecodeError as e:
                    raise IOError(
                        f"Index corrupted in line {lineno} of {req_index}: {e}\nContent: {line}"
                    )

    def add_examples(self, req: hexz_pb2.AddTrainingExamplesRequest) -> None:
        # Not all examples were necessarily generated using the same model key,
        # since model updates can happen at any time. Store the request under the
        # latest model that was used.
        if len(req.examples) == 0:
            return
        name = req.examples[-1].model_key.name
        checkpoint = req.examples[-1].model_key.checkpoint
        examples_dir = self._examples_dir(name, checkpoint)
        now = datetime.datetime.now(tz=pytz.UTC).strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(examples_dir, f"{now}.gz")
        with self.examples_lock:
            self._index_add(model_name=name, filename=filename, req=req)
            if not os.path.exists(examples_dir):
                os.makedirs(examples_dir)

        # Files can be written concurrently, as their names differ.
        with gzip.open(filename, "wb") as f:
            f.write(req.SerializeToString())
        with self.acquire_h5(name) as h:
            # Add examples to HDF5. This must happen sequentially, as vanilla HDF5 does
            # not support concurrent writes.
            boards = [
                torch.load(io.BytesIO(e.board), weights_only=True).numpy()
                for e in req.examples
            ]
            action_masks = [
                torch.load(io.BytesIO(e.action_mask), weights_only=True).numpy()
                for e in req.examples
            ]
            move_probs = [
                torch.load(io.BytesIO(e.move_probs), weights_only=True).numpy()
                for e in req.examples
            ]
            # Examples and model use float32, value must be of the same dtype.
            values = [np.array([e.result], dtype=np.float32) for e in req.examples]
            checkpoints = [np.array([e.model_key.checkpoint]) for e in req.examples]
            for label, data in zip(
                ["boards", "action_masks", "move_probs", "values", "checkpoints"],
                [boards, action_masks, move_probs, values, checkpoints],
            ):
                if label not in h:
                    chunk_batch = 128  # add this many examples to each HDF5 chunk
                    h.create_dataset(
                        label,
                        data=data,
                        compression="gzip",
                        chunks=(
                            chunk_batch,
                            *data[0].shape,
                        ),  # make sure to use efficient chunk sizes
                        maxshape=(None, *data[0].shape),
                    )
                else:
                    h[label].resize(h[label].shape[0] + len(data), axis=0)
                    h[label][-len(data) :] = data
            h.flush()  # Flush HDF5 file, since we keep it open.

    def find_requests(
        self,
        model_name: str,
        game_id: str,
        execution_id: str | None = None,
        max_results: int = 5,
    ) -> list[hexz_pb2.AddTrainingExamplesRequest]:
        """Finds requests matching the given query parameters in the index."""
        result = []
        model_base = self._model_base(model_name)
        with self.models_lock:
            iter = self._index_iter(model_name, game_id, execution_id)
            entries = list(itertools.islice(iter, max_results))
        for entry in entries:
            with gzip.open(os.path.join(model_base, entry.relpath), "rb") as f_in:
                data = f_in.read()
                req = hexz_pb2.AddTrainingExamplesRequest()
                req.ParseFromString(data)
                result.append(req)

        return result

    def recent_requests(
        self,
        model_name: str,
        max_results: int = 100,
    ) -> list[RequestIndexEntry]:
        """Returns RequestIndexEntry infos for the last max_results requests added to the index."""
        result = []
        lines = deque(maxlen=max_results)
        with self.models_lock:
            req_index = self._request_index_path(model_name)
            if not os.path.exists(req_index):
                return []
            for line in open(req_index, "r", encoding="utf-8"):
                if not line.strip():
                    continue
                lines.append(line)
        for line in lines:
            try:
                entry = json.loads(line)
                result.append(RequestIndexEntry(**entry))
            except json.JSONDecodeError as e:
                raise IOError(f"Index {req_index} corrupted: {e}\nContent: {line}")

        return result
