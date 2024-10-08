"""Test cases for the training.py module."""

import io
import torch
from pyhexz import hexz_pb2
from pyhexz.model import HexzNeuralNetwork
from pyhexz.modelrepo import LocalModelRepository
from pyhexz.training import HDF5Dataset


def _torch_bytes(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def test_hdf5dataset_iter(tmp_path):
    # Iterate over a HDF5Dataset.
    d = tmp_path / "repo"
    repo = LocalModelRepository(d)
    model = HexzNeuralNetwork()
    model_name = "testmodel"
    repo.store_model(model_name, 0, model)
    req = hexz_pb2.AddTrainingExamplesRequest(
        examples=[
            hexz_pb2.TrainingExample(
                encoding=hexz_pb2.TrainingExample.Encoding.PYTORCH,
                model_key=hexz_pb2.ModelKey(
                    name=model_name,
                    checkpoint=0,
                ),
                board=_torch_bytes(torch.randn((11, 11, 10))),
                action_mask=_torch_bytes(torch.randn((2, 11, 10))),
                move_probs=_torch_bytes(torch.randn((2, 11, 10))),
                result=0.5,
            ),
            hexz_pb2.TrainingExample(
                encoding=hexz_pb2.TrainingExample.Encoding.PYTORCH,
                model_key=hexz_pb2.ModelKey(
                    name=model_name,
                    checkpoint=0,
                ),
                board=_torch_bytes(torch.randn((11, 11, 10))),
                action_mask=_torch_bytes(torch.randn((2, 11, 10))),
                move_probs=_torch_bytes(torch.randn((2, 11, 10))),
                result=0.5,
            )
        ],
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
            assert action_mask.shape == (2, 11, 10)
            assert move_probs.shape == (2, 11, 10)
            assert value.shape == (1,)
    repo.close_all()
    
