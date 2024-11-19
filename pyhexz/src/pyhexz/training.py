from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import dataclasses
from datetime import datetime
import logging
import queue
import threading
import time
from typing import Iterator, Tuple
import grpc
import h5py
import numpy as np
import torch
from torch import nn

from pyhexz import hexz_pb2
from pyhexz.config import TrainingConfig
from pyhexz.modelrepo import LocalModelRepository, ModelRepository


@dataclass
class ModelInfo:
    """Same as hexz_pb2.ModelKey. Only used for info messages, logging, and such."""

    name: str
    checkpoint: int


@dataclass
class TrainingResultInfo:
    started: str  # YYYY-MM-DD HH:MM:SSZ
    done: str  # YYYY-MM-DD HH:MM:SSZ
    model: ModelInfo
    num_epochs: int
    training_time: float
    total_time: float
    examples_window: Tuple[int, int]


@dataclass
class TrainingStats:
    # Number of example requests received.
    example_requests: int = 0
    examples: int = 0
    training_runs_started: int = 0
    training_runs_completed: int = 0
    last_training_ex_count: int = 0
    last_training_info: TrainingResultInfo | None = None


def rchunks(start: int, end: int, size: int) -> list[slice]:
    return reversed([slice(max(start, i - size), i) for i in range(end, start, -size)])


class HDF5IterableDataset(torch.utils.data.IterableDataset):
    """PyTorch iterable-style Dataset implementation to read Hexz examples from HDF5."""

    def __init__(
        self, h5_file: h5py.File, window_size=0, shuffle=False, shuffle_chunk_size=2**20
    ):
        """Builds a new dataset that reads from the h5_file HDF5 file handle.

        Assumes that the dataset has exclusive access to the h5_file.

        Arguments:
            h5_file: the HDF5 file to read data from. Must have datasets for
                "boards", "action_masks", "move_probs", and "values" of the
                expected shapes.
            window_size: The maximum number of examples to use from the file.
                Only the rightmost (newest) window_size examples are used.
                None means that all examples in h5_file will be used.
            shuffle: If True, examples are shuffled within large (2^20) chunks.
                (We cannot shuffle the whole dataset unless it fits in memory.
                 Shuffling large chunks independently should be good enough.)
        """
        super().__init__()
        self.h5_file = h5_file
        self.shuffle = shuffle
        self.shuffle_chunk_size = shuffle_chunk_size
        l = len(h5_file["boards"])
        if window_size == 0 or window_size >= l:
            self.start = 0
        else:
            self.start = l - window_size
        self.end = l

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        rng = np.random.default_rng()
        size = self.shuffle_chunk_size if self.shuffle else 4096
        for c in rchunks(self.start, self.end, size):
            bs = self.h5_file["boards"][c]
            am = self.h5_file["action_masks"][c]
            mp = self.h5_file["move_probs"][c]
            vs = self.h5_file["values"][c]
            if self.shuffle:
                js = rng.permutation(len(bs))
            else:
                js = np.arange(len(bs))
            for j in js:
                yield ((bs[j], am[j]), (mp[j], vs[j]))


class TrainingTask:
    """TrainingTask is responsible for training the model given the stored examples.

    A task will typically be executed by an individual thread or in a thread pool.
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

    def execute(self) -> TrainingResultInfo:
        t_start = time.time()
        device = self.config.device
        model = self.model_repo.get_model(self.model_name, self.checkpoint)
        model.train()
        model = model.to(device)
        with self.model_repo.acquire_h5(self.model_name) as h:
            dataset = HDF5IterableDataset(
                h,
                window_size=self.config.training_examples_window_size,
                shuffle=self.config.shuffle,
            )
            self.logger.info(f"Training dataset size: {len(dataset)}")

            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.config.batch_size,
                shuffle=False,  # Shuffling happens in the HDF5IterableDataset
                pin_memory=self.config.pin_memory,
            )

            pr_loss_fn = nn.CrossEntropyLoss()
            val_loss_fn = nn.MSELoss()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.adam_weight_decay,
            )
            t_iter_start = time.time()
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
                    # Need to flatten y_pr, which is stored in (2, 11, 10) form in HDF5.
                    pr_loss = pr_loss_fn(pred_pr, y_pr.flatten(1))
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
        t_end = time.time()
        self.logger.info(
            f"_run_training done in {(t_end-t_start):.3f}s."
            f" setup time: {(t_iter_start-t_start):.3f}s."
            f" training time: {training_ns/1e9:.3f}s."
        )
        return TrainingResultInfo(
            model=ModelInfo(self.model_name, next_cp),
            started=datetime.fromtimestamp(t_start).isoformat(),
            done=datetime.fromtimestamp(t_end).isoformat(),
            total_time=t_end - t_start,
            num_epochs=self.config.num_epochs,
            training_time=training_ns / 1e9,
            examples_window=(dataset.start, dataset.end),
        )


class TrainingState:
    """TrainingState is the entry point for adding examples and training the model.

    A training server should maintain a single instance of TrainingState.
    Instances are thread-safe. All methods can be called from multiple threads.
    """

    def __init__(
        self,
        model_repo: LocalModelRepository,
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
        # Get number of examples from HDF5, so we can restart the training
        # server at any point without resetting the next training trigger.
        h5_examples = model_repo.h5_size(model_name)
        self._stats = TrainingStats(
            examples=h5_examples,
            last_training_ex_count=(
                h5_examples - h5_examples % config.training_trigger_threshold
            ),
        )
        self._latest_request: hexz_pb2.AddTrainingExamplesRequest = None
        self._executor = ThreadPoolExecutor(max_workers=8)
        self.is_training: bool = False
        self._subscriptions: list[queue.SimpleQueue] = []

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
            >= self._stats.last_training_ex_count
            + self.config.training_trigger_threshold
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
        if self._subscriptions:
            self.logger.info(
                f"Sending TrainingStarted control event to subscribed workers"
            )
            for q in self._subscriptions:
                q.put(
                    hexz_pb2.ControlEvent(
                        training_started=hexz_pb2.ControlEvent.TrainingStarted(),
                    )
                )
        self.logger.info(
            f"Starting a new TrainingTask at {self._stats.examples} examples"
        )
        fut = self._executor.submit(task.execute)
        fut.add_done_callback(self._training_done)
        self.is_training = True
        self._stats.last_training_ex_count = self._stats.examples
        self._stats.training_runs_started += 1

    def _training_done(self, fut: Future[TrainingResultInfo]):
        """A callback that is called when a TrainingTask is done."""
        with self.lock:
            self._stats.training_runs_completed += 1
            self.is_training = False
            res = fut.result()
            self.latest_checkpoint = res.model.checkpoint
            self._stats.last_training_info = res
            self.logger.info(f"Training done: {res}")
            # Trigger next training round immediately if there are enough
            # examples already.
            if self._should_train():
                self._start_training()
            else:
                # Notify workers to resume.
                for q in self._subscriptions:
                    q.put(
                        hexz_pb2.ControlEvent(
                            training_done=hexz_pb2.ControlEvent.TrainingDone(),
                        )
                    )

    def _add_examples(self, req: hexz_pb2.AddTrainingExamplesRequest):
        """Saves the examples from the given request in the repository.

        If enough new examples have been collected, a new training is started,
        unless one is already ongoing.
        """
        # t_start = time.perf_counter_ns()
        self.model_repo.add_examples(req)
        # t_end = time.perf_counter_ns()
        # self.logger.info(
        #     f"Stored {len(req.examples)} examples in the repo in {(t_end-t_start)/1e6:.0f}ms"
        # )
        with self.lock:
            self._latest_request = req
            self._stats.example_requests += 1
            self._stats.examples += len(req.examples)
            if self._should_train():
                self._start_training()

    def add_examples(self, req: hexz_pb2.AddTrainingExamplesRequest):
        """Asynchronously saves the examples from the request in the repository."""
        self._executor.submit(self._add_examples, req)

    def accept(self, req: hexz_pb2.AddTrainingExamplesRequest):
        """Check if the request should be accepted for training.
        
        Examples must be based on a recent checkpoint of the right model.
        """
        key = self.model_key()
        if len(req.examples) == 0:
            return False
        rkey = req.examples[-1].model_key
        if rkey.name != key.name:
            return False
        cp_age = key.checkpoint - rkey.checkpoint
        return cp_age >= 0 and cp_age < 3

    def stats(self) -> TrainingStats:
        """Returns a copy of the training stats."""
        with self.lock:
            return dataclasses.replace(self._stats)

    def latest_request(self) -> hexz_pb2.AddTrainingExamplesRequest:
        with self.lock:
            return self._latest_request

    # Sentinel used in subscription queues to signal end-of-stream.
    _END_OF_STREAM = object()

    def _unsubscribe_events(self, q: queue.SimpleQueue):
        with self.lock:
            try:
                q.put(self._END_OF_STREAM)
                self._subscriptions.remove(q)
            except ValueError:
                pass

    def subscribe_events(
        self, ctx: grpc.ServicerContext
    ) -> Iterator[hexz_pb2.ControlEvent]:
        """Called by gRPC clients to subscribe to training events.

        Arguments:
            ctx: the streaming RPC context of the caller. This method will register
              a callback to unsubscribe the caller when its RPC gets terminated.
        """
        q = queue.SimpleQueue()
        send_started = False
        with self.lock:
            self._subscriptions.append(q)
            if self.is_training:
                send_started = True
        ctx.add_callback(lambda: self._unsubscribe_events(q))
        if send_started:
            # Send an initial Pause event since we are in the middle of training.
            yield hexz_pb2.ControlEvent(
                training_started=hexz_pb2.ControlEvent.TrainingStarted(),
            )
        while ctx.is_active():
            event = q.get()
            if event is self._END_OF_STREAM:
                return
            yield event
