"""Test cases for the training.py module."""

from collections import defaultdict
import dataclasses
import logging
import time
import h5py
import io
import json
import numpy as np
import pytest
import torch
from pyhexz import hexz_pb2
from pyhexz.config import TrainingConfig
from pyhexz.experiment import initialize_training_experiment
from pyhexz.model import HexzNeuralNetwork
from pyhexz.modelrepo import LocalModelRepository
from pyhexz.training import HDF5IterableDataset, TrainingState, TrainingTask, rchunks


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


def test_hdf5_dataset_iter(tmp_path):
    # Iterate over a HDF5IterableDataset.
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


def test_shuffled_dataset_reads_reproducible_random_blocks(tmp_path):
    path = tmp_path / "examples.h5"
    with h5py.File(path, "w") as h:
        values = np.arange(32, dtype=np.float32)
        h.create_dataset("boards", data=values.reshape(32, 1))
        h.create_dataset("action_masks", data=values.reshape(32, 1))
        h.create_dataset("move_probs", data=values.reshape(32, 1))
        h.create_dataset("values", data=values.reshape(32, 1))

        first_dataset = HDF5IterableDataset(
            h,
            shuffle=True,
            shuffle_chunk_size=4,
            seed=7,
        )
        first_values = [
            int(board.item())
            for ((board, _), _), _index in zip(first_dataset, range(8))
        ]
        second_dataset = HDF5IterableDataset(
            h,
            shuffle=True,
            shuffle_chunk_size=4,
            seed=7,
        )
        second_values = [
            int(board.item())
            for ((board, _), _), _index in zip(second_dataset, range(8))
        ]

    assert first_values == second_values
    assert first_dataset.sampled_ranges == second_dataset.sampled_ranges
    # Every four yielded examples come from one contiguous read block, while
    # successive blocks come from distinct, randomly ordered replay regions.
    first_block = {value // 4 for value in first_values[:4]}
    second_block = {value // 4 for value in first_values[4:]}
    assert len(first_block) == 1
    assert len(second_block) == 1
    assert first_block != second_block


def test_gradient_statistics_count_each_parameter_once(tmp_path):
    model = HexzNeuralNetwork(blocks=1, filters=8)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    config = TrainingConfig(
        model_repo_base_dir=tmp_path,
        model_name="stats",
    )
    task = TrainingTask(
        "stats",
        checkpoint=0,
        model_repo=None,
        config=config,
        logger=logging.getLogger(__name__),
    )
    stats = defaultdict(
        lambda: {
            "parameters": torch.tensor(0.0),
            "gradients": torch.tensor(0.0),
        }
    )

    task.accumulate_stats(stats, model)

    expected_gradient_sq = sum(p.numel() for p in model.parameters())
    actual_gradient_sq = sum(v["gradients"].item() for v in stats.values())
    assert actual_gradient_sq == pytest.approx(expected_gradient_sq)
    assert set(stats) == {"Conv2d", "BatchNorm2d", "Linear"}


def test_training(tmp_path):
    num_requests = 1
    num_examples_per_request = 40
    d = tmp_path / "repo"
    repo = LocalModelRepository(d)
    model = HexzNeuralNetwork()
    model_name = "testmodel"
    config = TrainingConfig(
        model_repo_base_dir=d,
        model_name=model_name,
        device="cpu",
        num_epochs=1,
    )
    repo.store_model(model_name, 0, model)
    for i in range(num_requests):
        req = hexz_pb2.AddTrainingExamplesRequest(
            examples=[
                _training_example(model_name, checkpoint=0)
                for i in range(num_examples_per_request)
            ]
        )
        repo.add_examples(req)
    task = TrainingTask(
        model_name,
        checkpoint=0,
        model_repo=repo,
        config=config,
        logger=logging.getLogger(__name__),
    )
    result = task.execute()
    assert result.model.checkpoint == 1
    assert repo.get_latest_checkpoint(model_name) == result.model.checkpoint
    m = repo.get_model(model_name, result.model.checkpoint, repr="scriptmodule")
    assert m is not None


def test_training_batch_limit_and_optimizer_resume(tmp_path):
    d = tmp_path / "repo"
    repo = LocalModelRepository(d)
    model_name = "limited"
    repo.store_model(model_name, 0, HexzNeuralNetwork(blocks=1, filters=8))
    req = hexz_pb2.AddTrainingExamplesRequest(
        examples=[
            _training_example(model_name, checkpoint=0)
            for _ in range(24)
        ]
    )
    repo.add_examples(req)
    config = TrainingConfig(
        model_repo_base_dir=d,
        model_name=model_name,
        batch_size=4,
        num_epochs=7,
        training_batches_per_trigger=2,
        replay_sampling_chunk_size=2,
        training_seed=17,
        device="cpu",
    )

    first = TrainingTask(
        model_name,
        checkpoint=0,
        model_repo=repo,
        config=config,
        logger=logging.getLogger(__name__),
    ).execute()
    assert first.training_batches == 2
    assert first.examples_trained == 8
    assert first.global_step == 2
    assert first.replay_sample_seed == 17
    assert len(first.sampled_ranges) == 4
    assert first.setup_time >= 0
    assert first.data_loading_time >= 0
    assert first.device_transfer_time >= 0
    assert first.compute_time >= 0
    assert first.checkpoint_time >= 0
    state = repo.get_training_state(model_name, 1)
    assert state["optimizer_name"] == "adam"
    assert state["global_step"] == 2
    assert state["examples_trained"] == 8
    assert state["optimizer_state"]["state"]
    assert state["last_training_run"]["replay_sample_seed"] == 17
    assert len(state["last_training_run"]["sampled_ranges"]) == 4

    second = TrainingTask(
        model_name,
        checkpoint=1,
        model_repo=repo,
        config=config,
        logger=logging.getLogger(__name__),
    ).execute()
    assert second.training_batches == 2
    assert second.examples_trained == 8
    assert second.global_step == 4
    assert second.replay_sample_seed == 18
    state = repo.get_training_state(model_name, 2)
    assert state["global_step"] == 4
    assert state["examples_trained"] == 16


def test_training_state_honors_durable_checkpoint_limit(tmp_path):
    repo = LocalModelRepository(tmp_path)
    model_name = "bounded"
    repo.store_model(model_name, 0, HexzNeuralNetwork(blocks=1, filters=8))
    config = TrainingConfig(
        model_repo_base_dir=tmp_path,
        model_name=model_name,
        training_trigger_threshold=1,
        training_max_checkpoint=0,
    )
    unbounded = TrainingState(repo, model_name, logging.getLogger(__name__), config)
    assert not unbounded.status()["at_training_limit"]

    bounded_config = dataclasses.replace(config, training_max_checkpoint=1)
    below_limit = TrainingState(
        repo, model_name, logging.getLogger(__name__), bounded_config
    )
    assert not below_limit.status()["at_training_limit"]

    repo.store_model(model_name, 1, HexzNeuralNetwork(blocks=1, filters=8))
    at_limit = TrainingState(
        repo, model_name, logging.getLogger(__name__), bounded_config
    )
    status = at_limit.status()
    assert status["checkpoint"] == 1
    assert status["at_training_limit"]
    with at_limit.lock:
        assert not at_limit._should_train()


def test_training_rejects_non_finite_loss_before_checkpoint(tmp_path):
    repo = LocalModelRepository(tmp_path)
    model_name = "nonfinite"
    repo.store_model(
        model_name,
        0,
        HexzNeuralNetwork(blocks=1, filters=8),
    )
    example = _training_example(model_name=model_name)
    example.board = _torch_bytes(torch.full((11, 11, 10), torch.nan))
    repo.add_examples(
        hexz_pb2.AddTrainingExamplesRequest(examples=[example])
    )
    config = TrainingConfig(
        model_repo_base_dir=tmp_path,
        model_name=model_name,
        batch_size=1,
        num_epochs=1,
        training_batches_per_trigger=1,
        replay_sampling_chunk_size=1,
        device="cpu",
    )

    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        TrainingTask(
            model_name,
            checkpoint=0,
            model_repo=repo,
            config=config,
            logger=logging.getLogger(__name__),
        ).execute()

    assert repo.get_latest_checkpoint(model_name) == 0


def test_initialize_training_experiment_isolated_and_aligned(tmp_path):
    repo = LocalModelRepository(tmp_path)
    source_name = "source"
    repo.store_model(
        source_name,
        7,
        HexzNeuralNetwork(blocks=1, filters=8),
    )
    repo.add_examples(
        hexz_pb2.AddTrainingExamplesRequest(
            examples=[
                _training_example(source_name, checkpoint=7) for _ in range(12)
            ]
        )
    )
    repo.close_all()

    manifest = initialize_training_experiment(
        repo_base_dir=tmp_path,
        source_model=source_name,
        source_checkpoint=7,
        candidate_model="candidate-r4",
        replay_examples=10,
        trigger_threshold=4,
    )

    assert manifest["source"]["replay_range"] == [4, 12]
    assert manifest["training"]["requested_replay_examples"] == 10
    assert manifest["training"]["seed_replay_examples"] == 8
    assert repo.get_latest_checkpoint(source_name) == 7
    assert repo.h5_size(source_name) == 12
    assert repo.get_latest_checkpoint("candidate-r4") == 0
    assert repo.h5_size("candidate-r4") == 8
    with repo.acquire_h5(source_name) as source:
        with repo.acquire_h5("candidate-r4") as candidate:
            for name in ("boards", "action_masks", "move_probs", "values"):
                np.testing.assert_array_equal(candidate[name][:], source[name][-8:])
    with open(
        tmp_path / "models" / "flagz" / "candidate-r4" / "experiment.json"
    ) as f:
        assert json.load(f) == manifest
    repo.close_all()

    with pytest.raises(FileExistsError):
        initialize_training_experiment(
            repo_base_dir=tmp_path,
            source_model=source_name,
            source_checkpoint=7,
            candidate_model="candidate-r4",
            replay_examples=8,
            trigger_threshold=4,
        )


def test_initialize_training_experiment_with_scratch_model(tmp_path):
    repo = LocalModelRepository(tmp_path)
    repo.store_model("source", 7, HexzNeuralNetwork(blocks=1, filters=8))
    repo.add_examples(
        hexz_pb2.AddTrainingExamplesRequest(
            examples=[_training_example("source", checkpoint=7) for _ in range(8)]
        )
    )
    repo.close_all()

    manifest = initialize_training_experiment(
        repo_base_dir=tmp_path,
        source_model="source",
        source_checkpoint=7,
        candidate_model="scratch-rich",
        replay_examples=4,
        trigger_threshold=4,
        scratch_representation="rich_v1",
        scratch_blocks=2,
        scratch_filters=16,
        source_replay_end=6,
    )

    initialized = repo.get_model("scratch-rich", 0)
    assert initialized.ctor_args == {
        "blocks": 2,
        "filters": 16,
        "model_type": "resnet",
        "representation": "rich_v1",
    }
    assert manifest["candidate"]["initialization"]["kind"] == "random"
    assert manifest["source"]["replay_range"] == [2, 6]
    assert repo.h5_size("scratch-rich") == 4
    repo.close_all()
