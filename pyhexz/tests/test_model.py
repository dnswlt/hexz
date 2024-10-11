import pytest
import torch
from pyhexz.board import Board
from pyhexz.model import HexzCNNBlocks, HexzNeuralNetwork


def test_cnnblocks_trace():
    bs = HexzCNNBlocks()
    trace_input = torch.randint(0, 1, (16, *Board.shape), dtype=torch.float32)
    ts = torch.jit.trace(bs, trace_input)
    # Traced and "normal" model should yield same results.
    test_input = torch.randint(0, 1, (16, *Board.shape), dtype=torch.float32)
    torch.testing.assert_close(bs(test_input), ts(test_input))


@pytest.mark.parametrize("traced", [True, False])
def test_script_model(traced: bool):
    batch_size = 16
    model = HexzNeuralNetwork(traced=traced)
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


def test_traced_model_perf():
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
        return
    iterations = 16
    batch_size = 4096
    device = "cuda"

    model = torch.jit.script(HexzNeuralNetwork(traced=True))
    model.to(device)
    input = torch.rand((batch_size, *Board.shape)).to(device)
    action_mask = (torch.rand((batch_size, 2, 11, 10)) < 0.5).to(device)
    
    for i in range(iterations):
        move_probs, value = model(input, action_mask)
    
        assert move_probs.to("cpu").shape == (batch_size, 220)
        assert move_probs.dtype == torch.float32
        assert value.shape == (batch_size, 1)


