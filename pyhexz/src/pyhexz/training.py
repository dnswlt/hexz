"""Classes used for training in a training server."""

from collections import defaultdict
from collections.abc import Mapping
import logging
from google.protobuf.message import DecodeError
import io
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
        self.model_bytes = None
        self.model_bytes_version = ("undefined", -1)
        self.logger = logger
        self.stats = defaultdict(lambda: 0)
        self.started = time.time()
        self.training_time = 0

    def start_training_runner_task(self):
        t = TrainingRunnerTask(
            self.repo, self.model_key, self.config, self.batch, self.queue
        )
        t.start()
        self.logger.info("Started TrainingRunnerTask")
        self.training_runner_task = t
        self.state = self._STATE_TRAINING

    def handle_training_examples(self, msg: Mapping[str, Any]):
        req: hexz_pb2.AddTrainingExamplesRequest = msg["request"]
        reply_q = msg["reply_q"]
        batch = self.batch
        if self.state != self._STATE_ACCEPTING:
            resp = hexz_pb2.AddTrainingExamplesResponse(
                status=hexz_pb2.AddTrainingExamplesResponse.REJECTED_TRAINING,
                error_message=f"Server is currently not accepting examples",
            )
            reply_q.put(resp)
            return
        if req.model_key != self.model_key:
            resp = hexz_pb2.AddTrainingExamplesResponse(
                status=hexz_pb2.AddTrainingExamplesResponse.REJECTED_WRONG_MODEL,
                latest_model=self.model_key,
            )
            reply_q.put(resp)
            return
        resp = hexz_pb2.AddTrainingExamplesResponse(
            status=hexz_pb2.AddTrainingExamplesResponse.ACCEPTED,
            latest_model=self.model_key,
        )
        reply_q.put(resp)

        # Add examples and run training if we have a full batch.
        lim = min(len(req.examples), self.config.batch_size - len(batch))
        for ex in req.examples[:lim]:
            board = np.load(io.BytesIO(ex.board))
            pr = np.load(io.BytesIO(ex.move_probs))
            val = np.array([ex.result], dtype=np.float32)
            batch.append((board, (pr, val)))
        self.stats["examples"] += lim

        if len(batch) >= self.config.batch_size:
            self.start_training_runner_task()

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
        name = msg["name"]
        checkpoint = msg["checkpoint"]
        reply_q = msg["reply_q"]
        if (name, checkpoint) != (self.model_key.name, self.model_key.checkpoint):
            reply_q.put(None)
            return
        if self.model_bytes is None or self.model_bytes_version != (name, checkpoint):
            try:
                self.model_bytes = self.repo.get_model(name, checkpoint, as_bytes=True)
            except FileNotFoundError as e:
                self.logger.error(f"Cannot get model {name}:{checkpoint}: {e}")
                reply_q.put(None)
                return
            self.model_bytes_version = (name, checkpoint)
        reply_q.put(self.model_bytes)

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
            "current_batch_size": len(self.batch),
            "uptime_seconds": round(now - self.started, 3),
            "training_time_pct": round(self.training_time / (now - self.started) * 100, 2),
        }
        reply_q.put(info)

    def run(self):
        """This is the main dispatcher loop of the TrainingTask.
        All requests to this task must be sent via self.queue and are processed sequentially.
        """
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
