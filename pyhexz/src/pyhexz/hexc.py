from __future__ import annotations
import cython
from cython.cimports.libc.math import sqrt
import numpy as np
from typing import Optional

from pyhexz.board import CBoard


@cython.cclass
class CNN:
    _parent: Optional[CNN]
    _player: cython.int
    _move: tuple[cython.size_t, cython.size_t, cython.size_t, cython.float]
    _wins: cython.float
    _visit_count: cython.int
    _move_probs: Optional[np.ndarray]
    _children: list[CNN]

    def __init__(
        self, parent: Optional[CNN], player: int, move: tuple[int, int, int, float]
    ):
        self._parent = parent
        self._player = player
        self._move = move
        self._wins = 0.0
        self._visit_count = 0
        self._children = []
        self._move_probs = None

    def __repr__(self):
        c: CNN
        vc = [0] * len(self._children)
        qs = [0.0] * len(self._children)
        for i in range(len(self._children)):
            c = self._children[i]
            vc[i] = c._visit_count
            qs[i] = c._visit_count and c._wins / c._visit_count
        return f"CNN({self._visit_count=}, {self._move=}, {self._player=}, vc={vc}, qs={qs}, size={self.size()})"

    def size(self):
        n: CNN
        q = [self]
        s = 0
        while q:
            n = q.pop()
            s += 1
            q.extend(n._children)
        return s

    def set_move_probs(self, move_probs: np.ndarray) -> None:
        self._move_probs = move_probs
        assert self._move_probs.shape == (2, 11, 10)

    def player(self) -> int:
        return self._player

    def other_player(self) -> int:
        return 1 - self._player

    def clear_parent(self) -> None:
        self._parent = None

    def move(self) -> tuple[int, int, int, float]:
        return self._move

    def create_children(
        self, player: int, moves: list[tuple[int, int, int, float]]
    ) -> None:
        self._children = [None] * len(moves)
        cnn: CNN
        for i in range(len(moves)):
            cnn = CNN(self, player, moves[i])
            self._children[i] = cnn

    @cython.cfunc
    def puct(self) -> cython.float:
        uct_c: cython.float = 4  # Constant weight of the exploration term.
        assert self._parent._move_probs.shape == (2, 11, 10)
        typ: cython.size_t = self._move[0]
        r: cython.size_t = self._move[1]
        c: cython.size_t = self._move[2]
        pr: cython.float[:, :, :] = self._parent._move_probs
        if self._visit_count == 0:
            q = 0.0
        else:
            q = self._wins / self._visit_count
        return q + uct_c * pr[typ, r, c] * sqrt(self._parent._visit_count) / (
            1 + self._visit_count
        )

    def move_likelihoods(self, dtype=np.float32) -> np.ndarray:
        """Returns the move likelihoods for all children as a (2, 11, 10) ndarray.

        The ndarray indicates the likelihoods (based on visit count) for flags
        and normal moves. It sums to 1.
        """
        p = np.zeros((2, 11, 10), dtype=dtype)
        child: CNN
        for i in range(len(self._children)):
            child = self._children[i]
            typ, r, c, _ = child._move
            p[typ, r, c] = child._visit_count
        return p / p.sum()

    def best_child(self) -> Optional[CNN]:
        """Returns the best among all children.

        The best child is the one with the greatest visit count, a common
        choice in the MCTS literature.
        """
        if len(self._children) == 0:
            return None
        i: cython.size_t = 0
        cand_child: CNN
        best_child: CNN = self._children[0]
        for i in range(1, len(self._children)):
            cand_child = self._children[i]
            if cand_child._visit_count > best_child._visit_count:
                best_child = cand_child
        return best_child

    def backpropagate(self, result: float):
        r: cython.float = result
        node: CNN = self
        while node is not None:
            node._visit_count += 1
            if node._player == 0:
                node._wins += (result + 1) / 2
            else:
                node._wins += (-result + 1) / 2
            node = node._parent


def c_valid_idx(r_c):
    """Returns True if (r, c) = r_c represents a valid hexz board index."""
    r, c = r_c
    return r >= 0 and r < 11 and c >= 0 and c < 10 - r % 2


def c_neighbors_map() -> (
    dict[tuple[cython.size_t, cython.size_t], tuple[np.ndarray, np.ndarray]]
):
    """Returns a dict mapping all valid (r, c) indices to their neighbor indices.

    The neighbor indices are represented as (row, column) tuples."""
    result = {}
    for r in range(11):
        shift = r % 2  # Depending on the row, neighbors below and above are shifted.
        for c in range(10 - r % 2):
            ns = filter(
                c_valid_idx,
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


_C_NEIGHBORS = c_neighbors_map()


@cython.cfunc
def c_occupy_grass(
    board: CBoard, player: cython.int, r: cython.size_t, c: cython.size_t
):
    """Occupies the neighboring grass cells of move_idx (a 3-tuple index into a move) for player.

    Expects that the move has already been played.
    """
    b: cython.float[:, :, :] = board.b
    i: cython.size_t = 0
    x: cython.size_t = 0
    y: cython.size_t = 0

    nx, ny = _C_NEIGHBORS[(r, c)]
    for i in range(len(nx)):
        x = nx[i]
        y = ny[i]
        grass_val = b[8, x, y]
        if grass_val > 0 and grass_val <= b[1 + player * 4, r, c]:
            # Occupy: first remove grass
            b[8, x, y] = 0
            # the rest is exactly like playing a move.
            c_make_move(board, player, (1, r, c, grass_val))


@cython.ccall
def c_make_move(board: CBoard, player: int, move: tuple[int, int, int, float]):
    """Makes the given move.

    Args:
      board: the board as an (N, 11, 10) ndarray.
      player: 0 or 1
      move: a 4-tuple of (typ, r, c, val), where typ = 0 (flag) or 1 (normal)
    Does not check that it is a valid move. Should be called only
    with moves returned from `next_moves`.
    """
    b: cython.float[:, :, :] = board.b
    p4: cython.size_t = player * 4
    typ: cython.size_t = move[0]
    r: cython.size_t = move[1]
    c: cython.size_t = move[2]
    val: cython.float = move[3]
    next_val: cython.float = 0
    i: cython.size_t = 0
    nr: cython.size_t = 0
    nc: cython.size_t = 0
    nx: cython.long[:]
    ny: cython.long[:]

    b[typ + p4, r, c] = val
    played_flag = typ == 0
    # Block played cell for both players.
    b[2, r, c] = 1
    b[6, r, c] = 1
    # Set next value to 0 for occupied cell.
    b[3, r, c] = 0
    b[7, r, c] = 0
    # Block neighboring cells if a 5 was played.
    nx, ny = _C_NEIGHBORS[(r, c)]
    # Update next value of neighboring cells. If we played a flag, the next value is 1.
    if played_flag:
        next_val = 1
        board.nflags[player] -= 1
    else:
        next_val = val + 1
    if next_val <= 5:
        for nr, nc in zip(nx, ny):
            if b[2 + p4, nr, nc] == 0:
                if next_val > 5:
                    b[3 + p4, nr, nc] = 0
                if b[3 + p4, nr, nc] == 0:
                    b[3 + p4, nr, nc] = next_val
                elif b[3 + p4, nr, nc] > next_val:
                    b[3 + p4, nr, nc] = next_val
    else:
        for i in range(len(nx)):
            # Played a 5: block neighboring cells and clear next value.
            b[2 + p4, nx[i], ny[i]] = 1
            b[3 + p4, nx[i], ny[i]] = 0  # Clear next value.
    if not played_flag:
        c_occupy_grass(board, player, r, c)


def c_find_leaf(board: CBoard, n: CNN):
    i: cython.size_t = 0
    while n._children:
        c: CNN = n._children[0]
        best: CNN = n._children[0]
        best_uct: cython.float = -1
        for i in range(1, len(n._children)):
            c = n._children[i]
            u = c.puct()
            if u > best_uct:
                best = c
                best_uct = u
        c_make_move(board, best._player, best._move)
        n = best
    return n
