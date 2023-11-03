"""Classes used for training in a training server."""

from collections import defaultdict
from collections.abc import Mapping
import logging
import io
import typing
import numpy as np
import queue
import threading
import time
import torch
from torch import nn
from typing import Any, Optional
from pyhexz.config import TrainingConfig
from pyhexz import hexz_pb2
from pyhexz.errors import HexzError
from pyhexz.modelrepo import ModelRepository


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.examples = []

    def __getitem__(self, k):
        return self.examples[k]

    def __len__(self):
        return len(self.examples)

    def append(self, example):
        self.examples.append(example)

    def extend(self, examples):
        self.examples.extend(examples)


class TrainingRunnerTask(threading.Thread):
    def __init__(
        self,
        repo: ModelRepository,
        model_key: hexz_pb2.ModelKey,
        config: TrainingConfig,
        batch: InMemoryDataset,
        done_queue: queue.SimpleQueue,
    ):
        super().__init__()
        self.repo = repo
        self.model_key = model_key
        self.done_queue = done_queue
        self.config = config
        self.batch = batch

    def run(self):
        reply = {
            "type": "TrainingRunnerTask",
            "status": "ERROR",
        }
        try:
            started = time.perf_counter()
            self.run_training()
            reply["status"] = "OK"
            reply["model_key"] = self.model_key
        except Exception as e:
            reply["error"] = str(e)
        finally:
            elapsed = time.perf_counter() - started
            reply["elapsed"] = elapsed
            self.done_queue.put(reply)

    def run_training(self):
        batch = self.batch
        device = self.config.device
        model = self.repo.get_model(self.model_key.name, self.model_key.checkpoint)
        model.train()
        model = model.to(device)
        num_epochs = self.config.num_epochs
        batch_size = self.config.batch_size
        loader = torch.utils.data.DataLoader(
            dataset=batch,
            batch_size=batch_size,
            shuffle=self.config.shuffle,
            pin_memory=self.config.pin_memory,
        )

        pr_loss_fn = nn.CrossEntropyLoss()
        val_loss_fn = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ in range(num_epochs):
            for X, (y_pr, y_val) in loader:
                # Send to device.
                X = X.to(device)
                y_pr = y_pr.to(device)
                y_val = y_val.to(device)

                # Predict
                pred_pr, pred_val = model(X)

                # Compute loss
                pr_loss = pr_loss_fn(pred_pr.flatten(1), y_pr.flatten(1))
                val_loss = val_loss_fn(pred_val, y_val)
                loss = pr_loss + val_loss

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Training is done. Save model under incremented checkpoint.
        self.model_key.checkpoint += 1
        self.repo.store_model(
            name=self.model_key.name,
            checkpoint=self.model_key.checkpoint,
            model=model,
        )


class TimingStats:
    def __init__(self):
        self.sum_duration_micros: int = 0
        self.sum_duration_sq: float = 0.0
        self.min_duration_micros: int = 2**30
        self.max_duration_micros: int = 0
        self.count: int = 0

    def summarize(self):
        """Returns a dict with summary stats."""
        if self.count == 0:
            return {"count": 0}
        avg_duration = self.sum_duration_micros / 1e6 / self.count
        return {
            "count": self.count,
            "min_duration": self.min_duration_micros / 1e6,
            "max_duration": self.max_duration_micros / 1e6,
            "avg_duration": avg_duration,
            "std_duration": np.sqrt(
                self.sum_duration_sq / self.count - avg_duration**2
            ),
        }


class TrainingTask(threading.Thread):
    """TrainingTask runs as a separate thread, accepts training data and
    spaws a separate thread or process to train the model and save new
    checkpoints."""

    _STATE_ACCEPTING = "ACCEPTING"
    _STATE_TRAINING = "TRAINING"

    def __init__(
        self,
        repo: ModelRepository,
        config: TrainingConfig,
        queue: queue.SimpleQueue,
        logger: logging.Logger,
    ):
        super().__init__()
        self.repo = repo
        checkpoint = repo.get_latest_checkpoint(config.model_name)
        self.model_key = hexz_pb2.ModelKey(
            name=config.model_name,
            checkpoint=checkpoint,
        )
        self.queue = queue
        self.config = config
        self.batch = InMemoryDataset()
        self.training_runner_task: Optional[TrainingRunnerTask] = None
        self.state = self._STATE_ACCEPTING
        self.model_cache = {
            "model_key": None,
            "state_dict": None,
            "scriptmodule": None,
        }
        self.logger = logger
        self.stats = defaultdict(lambda: 0)
        self.timing_stats = TimingStats()
        self.started = time.time()
        self.training_time = 0

    def accept_model_key(self, model_key: hexz_pb2.ModelKey):
        return (
            model_key.name == self.model_key.name
            and model_key.checkpoint <= self.model_key.checkpoint
            and (self.model_key.checkpoint - model_key.checkpoint)
            <= self.config.max_checkpoint_diff
        )

    def capacity(self) -> int:
        return max(0, self.config.batch_size - len(self.batch))

    def start_training_runner_task(self):
        t = TrainingRunnerTask(
            self.repo, self.model_key, self.config, self.batch, self.queue
        )
        t.start()
        self.logger.info("Started TrainingRunnerTask")
        self.training_runner_task = t
        self.state = self._STATE_TRAINING
        # Start a new batch to be able to accept examples during training.
        self.batch = InMemoryDataset()

    def handle_training_examples(self, msg: Mapping[str, Any]):
        """Expects a AddTrainingExamplesRequest and adds it to the current batch.

        If the current batch is already full, and a training is still ongoing,
        the TrainingTask is at capacity and will respond 
        """
        req: hexz_pb2.AddTrainingExamplesRequest = msg["request"]
        reply_q = msg["reply_q"]
        if not self.accept_model_key(req.model_key):
            resp = hexz_pb2.AddTrainingExamplesResponse(
                status=hexz_pb2.AddTrainingExamplesResponse.REJECTED_WRONG_MODEL,
                latest_model=self.model_key,
            )
            reply_q.put(resp)
            return
        if self.capacity() == 0:
            # We are at capacity. Signal this to the client.
            self.logger.warning(f"TrainingTask is at capacity. Rejecting {len(req.examples)} examples.")
            resp = hexz_pb2.AddTrainingExamplesResponse(
                status=hexz_pb2.AddTrainingExamplesResponse.REJECTED_AT_CAPACITY,
                error_message=f"Server is currently not able to accept more examples",
            )
            reply_q.put(resp)
            return
        resp = hexz_pb2.AddTrainingExamplesResponse(
            status=hexz_pb2.AddTrainingExamplesResponse.ACCEPTED,
            latest_model=self.model_key,
        )
        reply_q.put(resp)

        # Translate examples.
        new_batch = []
        for ex in req.examples:
            # Update timing stats.
            self.timing_stats.count += 1
            self.timing_stats.min_duration_micros = min(
                self.timing_stats.min_duration_micros, ex.duration_micros
            )
            self.timing_stats.max_duration_micros = max(
                self.timing_stats.max_duration_micros, ex.duration_micros
            )
            self.timing_stats.sum_duration_micros += ex.duration_micros
            self.timing_stats.sum_duration_sq += (ex.duration_micros / 1e6) ** 2
            # Extract example data.
            if ex.encoding == hexz_pb2.TrainingExample.PYTORCH:
                board = torch.load(io.BytesIO(ex.board)).numpy()
                pr = torch.load(io.BytesIO(ex.move_probs)).numpy()
            else:
                board = np.load(io.BytesIO(ex.board))
                pr = np.load(io.BytesIO(ex.move_probs))
            if board.shape != (9, 11, 10):
                self.logger.error(
                    f"Received board with wrong shape: {board.shape}. Ignored."
                )
                return
            if pr.shape != (2, 11, 10):
                self.logger.error(
                    f"Received move_probs with wrong shape: {pr.shape}. Ignored."
                )
                return
            val = np.array([ex.result], dtype=np.float32)
            new_batch.append((board, (pr, val)))
        self.stats["examples"] += len(req.examples)
        lim = min(len(req.examples), self.capacity())
        # Add translated examples to batch and trigger a training run if we have a full batch.
        batch_add, batch_buf = new_batch[:lim], new_batch[lim:]
        self.batch.extend(batch_add)
        if len(self.batch) >= self.config.batch_size and self.state != self._STATE_TRAINING:
            # Only start a new training task if we're not already training.
            self.start_training_runner_task()
        cap = self.capacity()
        if cap > 0:
            # New capacity for the remaining batch_buf elements since we started a new training round.
            if len(batch_buf) >= cap:
                n_dropped = len(batch_buf) - (self.config.batch_size)
                batch_buf = batch_buf[:self.config.batch_size]
                self.logger.warning(f"Dropped {n_dropped} examples. Examples from request exceeded {self.config.batch_size=}.")
            self.batch.extend(batch_buf)

    def handle_training_result(self, msg: Mapping[str, Any]):
        # The training runner task must be done now.
        self.training_runner_task.join()
        self.stats["training_runs"] += 1
        self.training_time += msg.get("elapsed", 0)
        # Start a new batch, whether training was successful or not.
        self.batch = InMemoryDataset()
        self.state = self._STATE_ACCEPTING
        if msg["status"] == "OK":
            self.model_key = msg["model_key"]
            elapsed = msg["elapsed"]
            self.logger.info(
                f"Finished training batch of size {self.config.batch_size} for {self.config.num_epochs} epochs in {elapsed:.1f}s."
            )
            self.logger.info(
                f"Updated model key to {self.model_key.name}:{self.model_key.checkpoint}."
            )
        else:
            self.logger.info(f"Training failed: {msg.get('error')}")

    def handle_get_model_key(self, msg):
        reply_q = msg["reply_q"]
        reply_q.put(self.model_key)

    def handle_get_model(self, msg):
        model_key = msg.get("model_key", self.model_key)
        repr = msg["repr"]
        reply_q = msg["reply_q"]

        if model_key != self.model_key:
            reply_q.put({"error": "requested wrong model key"})
            return
        # Create a copy and work with it, so we can return it safely.
        model_key = hexz_pb2.ModelKey()
        model_key.CopyFrom(self.model_key)
        if self.model_cache["model_key"] != model_key:
            # Invalidate cache.
            self.model_cache.clear()
            self.model_cache["model_key"] = model_key

        cached = self.model_cache.get(repr)
        if cached is not None:
            reply_q.put({"data": cached, "model_key": model_key})
            return
        # Load from disk and cache.
        try:
            mbytes = self.repo.get_model(
                model_key.name, model_key.checkpoint, as_bytes=True, repr=repr
            )
            self.model_cache[repr] = mbytes
            reply_q.put({"data": mbytes, "model_key": model_key})
        except FileNotFoundError as e:
            self.logger.error(f"Cannot load model {model_key}: {e}")
            reply_q.put({"error": f"Cannot load model {model_key}"})

    def handle_get_training_info(self, msg):
        reply_q = msg["reply_q"]
        now = time.time()
        info = {
            "model_key": {
                "name": self.model_key.name,
                "checkpoint": self.model_key.checkpoint,
            },
            "config": self.config._asdict(),
            "state": self.state,
            "stats": dict(self.stats),
            "example_timing_stats": self.timing_stats.summarize(),
            "current_batch_size": len(self.batch),
            "uptime_seconds": round(now - self.started, 3),
            "training_time_pct": round(
                self.training_time / (now - self.started) * 100, 2
            ),
        }
        reply_q.put(info)

    def run(self):
        """This is the main dispatcher loop of the TrainingTask.
        All requests to this task must be sent via self.queue and are processed sequentially.
        """
        self.logger.info(
            f"TrainingTask started. Using model {self.model_key.name}:{self.model_key.checkpoint}"
        )
        while True:
            # We use bytes as input and output in the communication with this
            # TrainingTask to simplify serialization when running in a separate process.
            msg = self.queue.get()
            if msg["type"] == "TrainingRunnerTask":
                self.handle_training_result(msg)
            elif msg["type"] == "AddTrainingExamplesRequest":
                self.handle_training_examples(msg)
            elif msg["type"] == "GetModelKey":
                self.handle_get_model_key(msg)
            elif msg["type"] == "GetModel":
                self.handle_get_model(msg)
            elif msg["type"] == "GetTrainingInfo":
                self.handle_get_training_info(msg)
            else:
                raise HexzError(
                    f"TrainingTask: received unknown message type {msg['type']}"
                )
