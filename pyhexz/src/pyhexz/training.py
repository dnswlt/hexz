
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import dataclasses
import logging
import threading
import time
from typing import Any
import h5py
import torch

from pyhexz import hexz_pb2
from pyhexz.config import TrainingConfig
from pyhexz.modelrepo import LocalModelRepository, ModelRepository


@dataclass
class TrainingStats:
    example_requests: int = 0
    examples: int = 0
    training_runs: int = 0

    
class HDF5Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset implementation to read Hexz examples from HDF5."""

    def __init__(self, h5_file: h5py.File):
        """Builds a new dataset that reads from the .h5 file pointed to by path.
        
        The dataset will only read examples that existed at the time of instantiation.
        If more examples get added while this dataset is in use, they will be ignored
        by the dataset.
        """
        self.h5_file = h5_file
        self.size = len(self.h5_file["boards"])

    def __getitem__(self, k):
        """Returns a tuple of inputs and labels: ((board, action_mask), (move_probs, value)).
        
        The collation function of the PyTorch DataLoader handles this properly and returns a tuple
        of batches for the labels.
        """
        h = self.h5_file
        return ((h["boards"][k], h["action_masks"][k]), (h["move_probs"][k], h["values"][k]))

    def __len__(self):
        return self.size



# TODO: move this into TrainingTask.
# class TrainingRunnerTask(threading.Thread):
#     def __init__(
#         self,
#         repo: ModelRepository,
#         model_key: hexz_pb2.ModelKey,
#         config: TrainingConfig,
#         batch: InMemoryDataset,
#         done_queue: queue.SimpleQueue,
#         logger: logging.Logger,
#     ):
#         super().__init__()
#         self.repo = repo
#         self.model_key = model_key
#         self.done_queue = done_queue
#         self.config = config
#         self.batch = batch
#         self.logger = logger

#     def run(self):
#         reply = {
#             "type": "TrainingRunnerTask",
#             "status": "ERROR",
#         }
#         try:
#             started = time.perf_counter()
#             self.run_training()
#             reply["status"] = "OK"
#             reply["model_key"] = self.model_key
#         except Exception as e:
#             reply["error"] = str(e)
#         finally:
#             elapsed = time.perf_counter() - started
#             reply["elapsed"] = elapsed
#             self.done_queue.put(reply)

#     def run_training(self):
#         batch = self.batch
#         device = self.config.device
#         model = self.repo.get_model(self.model_key.name, self.model_key.checkpoint)
#         model.train()
#         model = model.to(device)
#         num_epochs = self.config.num_epochs
#         batch_size = self.config.batch_size
#         loader = torch.utils.data.DataLoader(
#             dataset=batch,
#             batch_size=batch_size,
#             shuffle=self.config.shuffle,
#             pin_memory=self.config.pin_memory,
#         )

#         pr_loss_fn = nn.CrossEntropyLoss()
#         val_loss_fn = nn.MSELoss()

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=self.config.learning_rate,
#             weight_decay=self.config.adam_weight_decay,
#         )

#         for epoch in range(num_epochs):
#             for i, ((X_board, X_action_mask), (y_pr, y_val)) in enumerate(loader):
#                 # Send to device.
#                 X_board = X_board.to(device)
#                 X_action_mask = X_action_mask.to(device)
#                 y_pr = y_pr.to(device)
#                 y_val = y_val.to(device)

#                 optimizer.zero_grad(set_to_none=True)

#                 # Predict
#                 pred_pr, pred_val = model(X_board, X_action_mask)

#                 # Compute loss
#                 pr_loss = pr_loss_fn(pred_pr.flatten(1), y_pr.flatten(1))
#                 val_loss = val_loss_fn(pred_val, y_val)
#                 loss = pr_loss + val_loss
#                 self.logger.info(
#                     f"Epoch {epoch}: pr_loss:{pr_loss.item():.3f} val_loss:{val_loss.item():.3f} loss: {loss.item():.3f}"
#                 )
#                 # Backpropagation
#                 loss.backward()
#                 optimizer.step()

#         # Training is done. Save model under incremented checkpoint.
#         self.model_key.checkpoint += 1
#         self.repo.store_model(
#             name=self.model_key.name,
#             checkpoint=self.model_key.checkpoint,
#             model=model,
#         )


class TrainingTask:
    """TrainingTask is responsible for training the model given the stored examples.

    Each task should be executed by an individual thread or in a thread pool.
    """

    def __init__(
        self,
        model_name: str,
        model_repo: LocalModelRepository,
        logger: logging.Logger,
    ):
        self.model_name = model_name
        self.model_repo = model_repo
        self.logger = logger

    def execute(self):
        self.logger.info(
            "Someone tried to run training. That's nice, but we don't have that implemented yet. I'll sleep for a bit, pretending to do work."
        )
        t_start = time.time()
        with self.model_repo.acquire_h5(self.model_name) as h:
            dataset = HDF5Dataset(h)
            self.logger.info(f"There are {len(dataset)} examples to process")
            time.sleep(1)
        self.logger.info(f"Training (fake) done after {time.time()-t_start:.3f}s")


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
        self.last_training = 0  # Example count when the last training was started.
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
            self._stats.examples >= self.last_training + self.config.training_trigger_threshold
            and not self.is_training
        )

    def _start_training(self):
        """Must only be called while holding the lock."""
        task = TrainingTask(self.model_name, self.model_repo, logger=self.logger)
        self.logger.info(
            f"Starting a new TrainingTask at {self._stats.examples} examples"
        )
        fut = self._executor.submit(task.execute)
        fut.add_done_callback(self.training_done)
        self.is_training = True
        self.last_training = self._stats.examples
        self._stats.training_runs += 1

    def _add_examples(self, req: hexz_pb2.AddTrainingExamplesRequest):
        """Saves the examples from the given request in the repository.

        If enough new examples have been collected, a new training is started,
        unless one is already ongoing.
        """
        self.model_repo.add_examples(req)
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

    def training_done(self, fut: Future[Any]):
        """A callback that is called when a TrainingTask is done."""
        self.logger.info("TrainingState.training_done")
        if fut.exception():
            self.logger.error(f"Training failed with exception: {fut.exception()}")
        with self.lock:
            self.is_training = False
            # Trigger next training round immediately if there are enough
            # examples already.
            if self._should_train():
                self._start_training()
