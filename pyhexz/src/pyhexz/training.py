from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import dataclasses
import logging
import threading
import time
from typing import Any
import h5py
import torch
from torch import nn

from pyhexz import hexz_pb2
from pyhexz.config import TrainingConfig
from pyhexz.modelrepo import LocalModelRepository, ModelRepository


@dataclass
class TrainingStats:
    example_requests: int = 0
    examples: int = 0
    training_runs_started: int = 0
    training_runs_completed: int = 0
    last_training: int = 0


class HDF5Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset implementation to read Hexz examples from HDF5."""

    def __init__(self, h5_file: h5py.File, window_size=None):
        """Builds a new dataset that reads from the .h5 file pointed to by path.

        Assumes that the dataset has exclusive access to the h5_file.

        Arguments:
            h5_file: the HDF5 file to read data from. Must have datasets for
                "boards", "action_masks", "move_probs", and "values" of the
                expected shapes.
            window_size: The maximum number of examples to use from the file.
                Only the rightmost (newest) window_size examples are used.
                None means that all examples in h5_file will be used.
        """
        self.h5_file = h5_file
        l = len(h5_file["boards"])
        if window_size is None or window_size >= l:
            self.start = 0
        else:
            self.start = l - window_size
        self.end = l

    def __getitem__(self, k):
        """Returns a tuple of inputs and labels: ((board, action_mask), (move_probs, value)).

        The collation function of the PyTorch DataLoader handles this properly and returns a tuple
        of batches for the labels.
        """
        h = self.h5_file
        i = self.start + k
        return (
            (h["boards"][i], h["action_masks"][i]),
            (h["move_probs"][i], h["values"][i]),
        )

    def __len__(self):
        return self.end - self.start


class TrainingTask:
    """TrainingTask is responsible for training the model given the stored examples.

    Each task should be executed by an individual thread or in a thread pool.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint: int,
        model_repo: LocalModelRepository,
        config: TrainingConfig,
        logger: logging.Logger,
    ):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.model_repo = model_repo
        self.config = config
        self.logger = logger

    def run_training(self):
        t_start = time.perf_counter_ns()
        device = self.config.device
        model = self.model_repo.get_model(self.model_name, self.checkpoint)
        model.train()
        model = model.to(device)
        with self.model_repo.acquire_h5(self.model_name) as h:
            dataset = HDF5Dataset(
                h, window_size=self.config.training_examples_window_size
            )
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
                pin_memory=self.config.pin_memory,
            )

            pr_loss_fn = nn.CrossEntropyLoss()
            val_loss_fn = nn.MSELoss()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.adam_weight_decay,
            )
            t_iter_start = time.perf_counter_ns()
            training_ns = 0
            for epoch in range(self.config.num_epochs):
                for (X_board, X_action_mask), (y_pr, y_val) in loader:
                    t_start_batch = time.perf_counter_ns()
                    # Send to device.
                    X_board = X_board.to(device)
                    X_action_mask = X_action_mask.to(device)
                    y_pr = y_pr.to(device)
                    y_val = y_val.to(device)

                    optimizer.zero_grad(set_to_none=True)

                    # Predict
                    pred_pr, pred_val = model(X_board, X_action_mask)

                    # Compute loss
                    pr_loss = pr_loss_fn(pred_pr.flatten(1), y_pr.flatten(1))
                    val_loss = val_loss_fn(pred_val, y_val)
                    loss = pr_loss + val_loss
                    self.logger.info(
                        f"Epoch {epoch}: pr_loss:{pr_loss.item():.3f} val_loss:{val_loss.item():.3f} loss: {loss.item():.3f}"
                    )
                    # Backpropagation
                    loss.backward()
                    optimizer.step()
                    t_end_batch = time.perf_counter_ns()
                    training_ns += t_end_batch - t_start_batch

        # Training is done. Save model under incremented checkpoint.
        next_cp = self.checkpoint + 1
        self.model_repo.store_model(
            name=self.model_name,
            checkpoint=next_cp,
            model=model,
        )
        t_end = time.perf_counter_ns()
        self.logger.info(
            f"run_training done in {(t_end-t_start)/1e9:.3f}s."
            f" setup time: {(t_iter_start-t_start)/1e9:.3f}s."
            f" training time: {training_ns/1e9:.3f}s."
        )
        return next_cp

    def execute(self):
        t_start = time.time()
        next_cp = self.run_training()
        t_end = time.time()
        self.logger.info(
            f"Training done after {t_end-t_start:.3f}s. Stored new model {self.model_name}:{next_cp}"
        )
        return next_cp


class TrainingState:
    """TrainingState is the entry point for adding examples and training the model.

    A training server should maintain a single instance of TrainingState.
    Instances are thread-safe. All methods can be called from multiple threads.

    Attributes:
        training_examples (int): Number of examples to collect before starting a
            new training run.
    """

    def __init__(
        self,
        model_repo: ModelRepository,
        model_name: str,
        logger: logging.Logger,
        config: TrainingConfig,
    ):
        self.lock = threading.Lock()
        self.model_repo = model_repo
        self.model_name = model_name
        self.logger = logger
        self.config = config
        self.latest_checkpoint = model_repo.get_latest_checkpoint(model_name)
        self._stats = TrainingStats()
        self._latest_request: hexz_pb2.AddTrainingExamplesRequest = None
        self._executor = ThreadPoolExecutor(max_workers=8)
        self.is_training: bool = False

    def model_key(self):
        """Returns the latest model key."""
        with self.lock:
            return hexz_pb2.ModelKey(
                name=self.model_name, checkpoint=self.latest_checkpoint
            )

    def _should_train(self) -> bool:
        """Must only be called while holding the lock."""
        return (
            self._stats.examples
            >= self._stats.last_training + self.config.training_trigger_threshold
            and not self.is_training
        )

    def _start_training(self):
        """Must only be called while holding the lock."""
        task = TrainingTask(
            self.model_name,
            self.latest_checkpoint,
            self.model_repo,
            config=self.config,
            logger=self.logger,
        )
        self.logger.info(
            f"Starting a new TrainingTask at {self._stats.examples} examples"
        )
        fut = self._executor.submit(task.execute)
        fut.add_done_callback(self.training_done)
        self.is_training = True
        self._stats.last_training = self._stats.examples
        self._stats.training_runs_started += 1

    def _add_examples(self, req: hexz_pb2.AddTrainingExamplesRequest):
        """Saves the examples from the given request in the repository.

        If enough new examples have been collected, a new training is started,
        unless one is already ongoing.
        """
        t_start = time.perf_counter_ns()
        self.model_repo.add_examples(req)
        t_end = time.perf_counter_ns()
        self.logger.info(f"Stored {len(req.examples)} examples in the repo in {(t_end-t_start)/1e6:.0f}ms")
        with self.lock:
            self._latest_request = req
            self._stats.example_requests += 1
            self._stats.examples += len(req.examples)
            if self._should_train():
                self._start_training()

    def add_examples(self, req: hexz_pb2.AddTrainingExamplesRequest):
        """Asynchronously saves the examples from the request in the repository."""
        self._executor.submit(self._add_examples, req)

    def stats(self):
        """Returns a copy of the training stats."""
        with self.lock:
            return dataclasses.replace(self._stats)

    def latest_request(self) -> hexz_pb2.AddTrainingExamplesRequest:
        with self.lock:
            return self._latest_request

    def training_done(self, fut: Future[int]):
        """A callback that is called when a TrainingTask is done."""
        with self.lock:
            self._stats.training_runs_completed += 1
            self.is_training = False
            self.latest_checkpoint = fut.result()
            # Trigger next training round immediately if there are enough
            # examples already.
            if self._should_train():
                self._start_training()
