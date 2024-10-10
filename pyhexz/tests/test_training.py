"""Test cases for the training.py module."""

import logging
import time
import h5py
import io
import numpy as np
import pytest
import torch
from pyhexz import hexz_pb2
from pyhexz.config import TrainingConfig
from pyhexz.model import HexzNeuralNetwork
from pyhexz.modelrepo import LocalModelRepository
from pyhexz.training import HDF5Dataset, HDF5IterableDataset, TrainingTask, rchunks


def _torch_bytes(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


@pytest.mark.skip(reason="only run to check performance")
def test_iterds_h5_perf():
    with h5py.File("/tmp/hexz-models/models/flagz/edgar/h5/examples.h5", "r") as h:
        t_start = time.perf_counter_ns()
        ds = HDF5IterableDataset(h)
        sum = 0
        for item in ds:
            sum += len(item)
        print(f"Took {(time.perf_counter_ns() - t_start)/1000000:.0f}ms")


@pytest.mark.skip(reason="only run to check performance")
def test_iterds_shuffled_h5_perf():
    with h5py.File("/tmp/hexz-models/models/flagz/edgar/h5/examples.h5", "r") as h:
        t_start = time.perf_counter_ns()
        ds = HDF5IterableDataset(h, shuffle=True)
        sum = 0
        for item in ds:
            sum += len(item)
        print(f"Took {(time.perf_counter_ns() - t_start)/1000000:.0f}ms")


def _training_example(model_name="test", checkpoint=0):
    return hexz_pb2.TrainingExample(
        encoding=hexz_pb2.TrainingExample.Encoding.PYTORCH,
        model_key=hexz_pb2.ModelKey(
            name=model_name,
            checkpoint=checkpoint,
        ),
        board=_torch_bytes(torch.randn((11, 11, 10))),
        action_mask=_torch_bytes(torch.randn((2, 11, 10)) < 0.5),  # boolean tensor
        move_probs=_torch_bytes(torch.randn((2, 11, 10))),
        result=0.5,
    )


def test_hdf5dataset(tmp_path):
    # Iterate over a HDF5Dataset.
    d = tmp_path / "repo"
    repo = LocalModelRepository(d)
    model = HexzNeuralNetwork()
    model_name = "testmodel"
    repo.store_model(model_name, 0, model)
    req = hexz_pb2.AddTrainingExamplesRequest(
        examples=[
            _training_example(model_name),
            _training_example(model_name),
        ]
    )
    repo.add_examples(req)
    # Closing the repo is not strictly necessary here, but this way
    # we also test that reading data after closing HDF5 handles is
    # still possible (they get re-opened).
    repo.close_all()
    with repo.acquire_h5(model_name) as h:
        dataset = HDF5Dataset(h)
        assert len(dataset) == len(req.examples)
        for (board, action_mask), (move_probs, value) in dataset:
            assert board.shape == (11, 11, 10)
            assert board.dtype == np.float32
            assert action_mask.shape == (2, 11, 10)
            assert action_mask.dtype == bool
            assert move_probs.shape == (2, 11, 10)
            assert move_probs.dtype == np.float32
            assert value.shape == (1,)
            assert value.dtype == np.float32
    repo.close_all()


def test_hdf5dataset_iter(tmp_path):
    # Iterate over a HDF5Dataset.
    d = tmp_path / "repo"
    repo = LocalModelRepository(d)
    model = HexzNeuralNetwork()
    model_name = "testmodel"
    repo.store_model(model_name, 0, model)
    req = hexz_pb2.AddTrainingExamplesRequest(
        examples=[
            _training_example(model_name),
            _training_example(model_name),
        ]
    )
    repo.add_examples(req)
    # Closing the repo is not strictly necessary here, but this way
    # we also test that reading data after closing HDF5 handles is
    # still possible (they get re-opened).
    repo.close_all()
    with repo.acquire_h5(model_name) as h:
        dataset = HDF5IterableDataset(h, shuffle=True)
        k = 0
        for (board, action_mask), (move_probs, value) in dataset:
            assert board.shape == (11, 11, 10)
            assert board.dtype == np.float32
            assert action_mask.shape == (2, 11, 10)
            assert action_mask.dtype == bool
            assert move_probs.shape == (2, 11, 10)
            assert move_probs.dtype == np.float32
            assert value.shape == (1,)
            assert value.dtype == np.float32
            k += 1
        assert k == len(req.examples)
    repo.close_all()


def test_rev_chunks():
    assert list(rchunks(0, 1, 1)) == [slice(0, 1)]
    assert list(rchunks(0, 2, 1)) == [slice(0, 1), slice(1, 2)]
    assert list(rchunks(0, 10, 5)) == [slice(0, 5), slice(5, 10)]
    assert list(rchunks(0, 10, 3)) == [
        slice(0, 1),
        slice(1, 4),
        slice(4, 7),
        slice(7, 10),
    ]
    assert list(rchunks(0, 10, 1000)) == [slice(0, 10)]


def test_training(tmp_path):
    num_examples = 8
    d = tmp_path / "repo"
    repo = LocalModelRepository(d)
    model = HexzNeuralNetwork()
    model_name = "testmodel"
    repo.store_model(model_name, 0, model)
    req = hexz_pb2.AddTrainingExamplesRequest(
        examples=[
            _training_example(model_name, checkpoint=0) for i in range(num_examples)
        ]
    )
    repo.add_examples(req)
    config = TrainingConfig(
        model_repo_base_dir=d,
        model_name=model_name,
        device="cpu",
        num_epochs=1,
    )
    task = TrainingTask(
        model_name,
        checkpoint=0,
        model_repo=repo,
        config=config,
        logger=logging.getLogger(__name__),
    )
    next_cp = task.execute()
    assert next_cp == 1
    assert repo.get_latest_checkpoint(model_name) == next_cp
    m = repo.get_model(model_name, next_cp, repr="scriptmodule")
    assert m is not None
