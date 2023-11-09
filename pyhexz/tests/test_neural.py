from pyhexz.hexc import CBoard
from pyhexz.hexz import NeuralMCTS, HexzNeuralNetwork

class TestNeuralMCTS:

    def test_play_game(self):
        b = CBoard()
        model = HexzNeuralNetwork()
        m = NeuralMCTS(model, device='cpu')
        examples = m.play_game(b, runs_per_move=100)
        assert len(examples) > 0
        assert all(e.result is not None for e in examples)
