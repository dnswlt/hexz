from typing import Tuple
import pytest
import torch
from pyhexz.board import Board
from pyhexz.model import CNNLayer, HexzNeuralNetwork, ResidualLayer
import torch.nn.functional as F


def _random_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
    board = torch.rand((1, 11, 11, 10), dtype=torch.float32)
    # Boolean tensor
    action_mask = torch.rand((1, 2, 11, 10)) > 0.5
    return (board, action_mask)


def test_conv2d():
    # Validate that inputs and outputs have the expected shapes.
    model = HexzNeuralNetwork(model_type="conv2d")
    assert isinstance(model.torso, CNNLayer)
    policy, value = model(*_random_inputs())
    assert value.shape == (1, 1)
    v = value[0].item()
    assert -1 <= v <= 1
    assert policy.shape == (1, 2 * 11 * 10)
    p = F.softmax(policy, dim=1)
    s = torch.sum(p).item()
    assert 0 - 1e-4 <= s <= 1 + 1e-4


def test_resnet():
    model = HexzNeuralNetwork(model_type="resnet")
    assert isinstance(model.torso, ResidualLayer)
    policy, value = model(*_random_inputs())
    assert value.shape == (1, 1)
    v = value[0].item()
    assert -1 <= v <= 1
    assert policy.shape == (1, 2 * 11 * 10)
    p = F.softmax(policy, dim=1)
    s = torch.sum(p).item()
    assert 0 - 1e-4 <= s <= 1 + 1e-4


def test_script_model():
    batch_size = 16
    model = HexzNeuralNetwork()
    scripted = torch.jit.script(model)

    input = torch.rand((16, *Board.shape))
    action_mask = torch.rand((batch_size, 2, 11, 10)) < 0.5
    move_probs, value = scripted(input, action_mask)
    assert move_probs.shape == (batch_size, 220)
    assert move_probs.dtype == torch.float32
    assert value.shape == (batch_size, 1)
    # Check that scripted model yields the same results.
    mp2, v2 = model(input, action_mask)
    torch.testing.assert_close(mp2, move_probs)
    torch.testing.assert_close(v2, value)
