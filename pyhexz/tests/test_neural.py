from pyhexz.board import Board
from pyhexz.hexz import NeuralMCTS, HexzNeuralNetwork

class TestNeuralMCTS:

    def test_play_game(self):
        b = Board()
        model = HexzNeuralNetwork()
        m = NeuralMCTS(b, model, device='cpu')
        examples = m.play_game(runs_per_move=100)
        assert len(examples) > 0
        assert all(e.result is not None for e in examples)
