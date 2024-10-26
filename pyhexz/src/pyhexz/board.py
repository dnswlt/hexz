"""Pure Python implementation of the Flagz board game."""

import numpy as np


def _init_neighbors_map():
    """Returns a dict mapping all valid (r, c) indices to their neighbor indices.

    The neighbor indices are represented as (row, column) tuples."""

    def _valid_idx(r_c):
        r, c = r_c
        return r >= 0 and r < 11 and c >= 0 and c < 10 - r % 2

    result = {}
    for r in range(11):
        shift = r % 2  # Depending on the row, neighbors below and above are shifted.
        for c in range(10 - r % 2):
            ns = filter(
                _valid_idx,
                [
                    (r, c + 1),
                    (r - 1, c + shift),
                    (r - 1, c - 1 + shift),
                    (r, c - 1),
                    (r + 1, c - 1 + shift),
                    (r + 1, c + shift),
                ],
            )
            nr, nc = zip(*ns)  # unzip
            result[(r, c)] = (np.array(nr), np.array(nc))

    return result


class Board:
    """Numpy representation of a hexz board.

    A board is represented by an (11, 11, 10) numpy array. Each 11x10 channel is
    a one-hot encoding of the presence of specific type of piece/obstacle/etc.
    The channels are:

    * 0: flags by P1
    * 1: cell value 1-5 for P1
    * 2: cells blocked for P1 (any occupied cell or a cell next to a 5)
    * 3: next value for P1
    * 4: number of remaining flags for P1
    * 5: flags by P2
    * 6: cell value 1-5 for P2
    * 7: cells blocked for P2
    * 8: next value for P2
    * 9: number of remaining flags for P1
    * 10: grass cells with value 1-5

    An action is specified by a (2, 11, 10) numpy array. The first 11x10 channel
    represents a flag move, the second one represents a regular cell move. A
    flag move must have a single 1 set, a normal move must have a single value
    1-5 set.
    """

    shape = (11, 11, 10)

    # Used to quickly get the indices of neighbor cells.
    neighbors = _init_neighbors_map()

    def __init__(self, other=None, dtype=np.float32, board=None):
        """Generates a new randomly initialized board or returns a copy of other, if set."""
        if other is not None:
            self.b = other.b.copy()
            return
        if board is not None:
            self.b = board
            return
        self.b = np.zeros(self.shape, dtype=dtype)
        # 3 flags for each player
        self.b[4] = 3
        self.b[9] = 3
        # Even rows have 10 cells, odd rows only 9, so mark the last cell in odd rows as blocked for P1+P2.
        self.b[2, [1, 3, 5, 7, 9], 9] = 1
        self.b[7, [1, 3, 5, 7, 9], 9] = 1
        # 2-tuple of valid indices in each slice.
        free_cells = (1 - self.b[2]).nonzero()
        # 15 randomly placed stones.
        rng = np.random.default_rng()
        stones = rng.choice(np.arange(0, len(free_cells[0])), replace=False, size=15)
        self.b[2, free_cells[0][stones], free_cells[1][stones]] = 1
        self.b[7, free_cells[0][stones], free_cells[1][stones]] = 1
        free_cells = (1 - self.b[2]).nonzero()
        # 5 grass cells
        grass = rng.choice(np.arange(0, len(free_cells[0])), replace=False, size=5)
        self.b[10, free_cells[0][grass], free_cells[1][grass]] = [1, 2, 3, 4, 5]
        self.b[2, free_cells[0][grass], free_cells[1][grass]] = 1
        self.b[7, free_cells[0][grass], free_cells[1][grass]] = 1

    @classmethod
    def from_numpy(cls, board):
        if board.shape != cls.shape:
            raise ValueError("Invalid board shape: " + str(board.shape))
        return cls(board=board)

    def b_for(self, player):
        """Returns the underlying ndarray representing the board, oriented for the given player.
        Only returns a copy if the board is not already oriented for the given player.
        """
        if player == 0:
            return self.b
        b = self.b.copy()
        b[0:5], b[5:10] = self.b[5:10], self.b[0:5]
        return b

    def score(self):
        """Returns the current score as a 2-tuple."""
        return (self.b[1].sum(), self.b[6].sum())

    def result(self):
        """Returns the final result of the board.

        1 (player 0 won), 0 (draw), -1 (player 1 won).
        """
        p0, p1 = self.score()
        return np.sign(p0 - p1)

    def flags(self, player):
        return int(self.b[4 + player * 5, 0, 0])
    
    def make_move(self, player, move):
        """Makes the given move.

        Args:
          player: 0 or 1
          move: a 4-tuple of (typ, r, c, val), where typ = 0 (flag) or 1 (normal)
        Does not check that it is a valid move. Should be called only
        with moves returned from `next_moves`.
        """
        typ, r, c, val = move
        b = self.b
        b[typ + player * 5, r, c] = val
        played_flag = typ == 0
        # Block played cell for both players.
        b[2, r, c] = 1
        b[7, r, c] = 1
        # Set next value to 0 for occupied cell.
        b[3, r, c] = 0
        b[8, r, c] = 0
        # Block neighboring cells if a 5 was played.
        nx, ny = Board.neighbors[(r, c)]
        # Update next value of neighboring cells. If we played a flag, the next value is 1.
        if played_flag:
            next_val = 1
            b[4 + player * 5] -= 1
        else:
            next_val = val + 1
        if next_val <= 5:
            for nr, nc in zip(nx, ny):
                if b[2 + player * 5, nr, nc] == 0:
                    # Cell is not blocked yet.
                    if b[3 + player * 5, nr, nc] == 0:
                        b[3 + player * 5, nr, nc] = next_val
                    elif b[3 + player * 5, nr, nc] > next_val:
                        b[3 + player * 5, nr, nc] = next_val
        else:
            # Played a 5: block neighboring cells and clear next value.
            b[2 + player * 5, nx, ny] = 1
            b[3 + player * 5, nx, ny] = 0  # Clear next value.

        # Occupy neighboring grass cells.
        if not played_flag:
            self.occupy_grass(player, r, c)

    def occupy_grass(self, player, r, c):
        """Occupies the neighboring grass cells of move_idx (a 3-tuple index into a move) for player.

        Expects that the move has already been played.
        """
        nx, ny = Board.neighbors[(r, c)]
        for i, j in zip(nx, ny):
            grass_val = self.b[10, i, j]
            if grass_val > 0 and grass_val <= self.b[1 + player * 5, r, c]:
                # Occupy: first remove grass
                self.b[10, i, j] = 0
                # the rest is exactly like playing a move.
                self.make_move(player, (1, r, c, grass_val))

    def next_moves(self, player):
        """Returns all possible next moves.

        A move is represented as a (2, 11, 10) ndarray. The first slice represents
        flag moves, the second one represents normal moves. A flag move will have exactly
        one element set to 1 in slice 0. A normal move will have exactly one element set to
        1-5 in slice 1.
        """
        moves = []
        # Do we have unoccupied cells and flags left? Then we can place another one.
        if self.b[4 + player * 5, 0, 0] > 0:
            # Flag in any unoccupied cell.
            rs, cs = np.nonzero(
                self.b[2 + player * 5] == 0
            )  # funky way to get indices for all free cells.
            moves.extend((0, r, c, 1) for r, c in zip(rs, cs))
        # Collect all cells with a non-zero next value.
        rs, cs = np.nonzero(self.b[3 + player * 5])
        moves.extend((1, r, c, self.b[3 + player * 5, r, c]) for r, c in zip(rs, cs))
        return moves

    def validate(self) -> None:
        """Checks that this instance represents a valid board.

        Raises an exeption if that is not the case.
        """
        b = self.b
        if np.any(b[2, [1, 3, 5, 7, 9], 9] == 0) or np.any(b[7, [1, 3, 5, 7, 9], 9] == 0) :
            raise ValueError("Invalid cells are not marked as blocked")
        if np.any(b[1] * b[6]):
            raise ValueError("Both players have a value in the same cell.")
        if np.any(b[0] * b[5]):
            raise ValueError("Both players have a flag in the same cell.")
        if np.any(b[2] * (b[0] + b[1]) != (b[0] + b[1])):
            raise ValueError("Not all occupied cells are marked as blocked for P1.")
        if np.any(b[7] * (b[5] + b[6]) != (b[5] + b[6])):
            raise ValueError("Not all occupied cells are marked as blocked for P2.")
        if np.any(b[0] * b[1]):
            raise ValueError("P1 has a value in a cell with a flag.")
        if np.any(b[5] * b[6]):
            raise ValueError("P2 has a value in a cell with a flag.")
        if np.any(b[3] * (b[0] + b[1] + b[2])):
            raise ValueError("P1 has a next value in an occupied cell.")
        if np.any(b[8] * (b[5] + b[6] + b[7])):
            raise ValueError("P2 has a next value in a blocked cell.")
        grass_values, grass_counts = np.unique(b[10], return_counts=True)
        if not (set(grass_values) <= set([0, 1, 2, 3, 4, 5])):
            raise ValueError(f"Wrong values for grass cells: {grass_values}")
        if len(grass_counts) > 1 and grass_counts[1:].max() > 1:
            raise ValueError(f"Duplicate grass cells: {grass_values}, {grass_counts}")
        if b.max() > 5:
            raise ValueError("Max value is > 5.")
        if b.min() < 0:
            raise ValueError("Min value is < 0.")

