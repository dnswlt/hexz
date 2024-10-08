"""Test cases for the board.py module."""

import numpy as np
import pytest

from pyhexz.board import Board

class TestBoard:

    def test_create(self):
        b = Board()
        assert b.flags(0) == 3
        assert b.flags(1) == 3

    def test_validate(self):
        b = Board()
        b.validate()

    def test_copy(self):
        b = Board()
        c = Board(b)
        assert b.b is not c.b
        assert np.all(b.b == c.b)

    def test_from_numpy(self):
        b = Board.from_numpy(np.zeros((11, 11, 10)))
        with pytest.raises(ValueError):
            b.validate()  # Cannot be valid, it's all zeroes.
        
    def test_b_for(self):
        b = Board()
        assert b.b_for(player=0) is b.b, "Should not copy for player 0"
        c = Board.from_numpy(b.b_for(player=1)).b_for(player=1)
        assert np.all(b.b == c), "Flipping twice should be the identity"

    def test_play_game(self):
        """Play one full game, always making the first of the possible next moves."""
        b = Board()
        player = 0
        num_moves = 0
        moves = b.next_moves(player)
        while moves and num_moves < 200:
            b.make_move(player, moves[0])
            num_moves += 1
            player = 1 - player
            moves = b.next_moves(player)
            if not moves:
                player = 1 - player
                moves = b.next_moves(player)
        assert 5 < num_moves < 200
        s = b.score()
        assert isinstance(s, tuple)
        assert len(s) == 2
        assert s[0] + s[1] > 0
        assert b.flags(0) == 0
        assert b.flags(1) == 0
