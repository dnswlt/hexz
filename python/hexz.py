"""Python implementation of the Flagz game.

This file includes the game, MCTS, and neural network implementations.
"""

import argparse
from bisect import bisect_left
import collections
from contextlib import contextmanager
from functools import wraps
import glob
import h5py
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import sys
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import Optional
from uuid import uuid4

import hexc

# Stores accumulated running time in micros and call counts of functions annotated with @timing.
_PERF_ACC_MICROS = collections.Counter()
_PERF_COUNTS = collections.Counter()
_PERF_TIMING_ENABLED = True


def disable_perf_stats():
    global _PERF_TIMING_ENABLED
    _PERF_TIMING_ENABLED = False


def timing(f):
    if not _PERF_TIMING_ENABLED:
        return f

    @wraps(f)
    def wrap(*args, **kw):
        t_start = time.perf_counter_ns()
        result = f(*args, **kw)
        t_end = time.perf_counter_ns()
        name = f.__qualname__
        _PERF_ACC_MICROS[name] += (t_end - t_start) // 1000
        _PERF_COUNTS[name] += 1
        return result

    return wrap


@contextmanager
def print_time(name):
    """Context manager to print the time a code block took to execute."""
    t_start = time.perf_counter_ns()
    yield
    elapsed = time.perf_counter_ns() - t_start
    print(f"{name} took {int(elapsed/1e6)} ms")


@contextmanager
def timing_ctx(name):
    """Context manager to time a block of code. While @timing can only be used on functions,
    this can be used on any block of code.
    """
    if not _PERF_TIMING_ENABLED:
        yield
        return
    t_start = time.perf_counter_ns()
    yield
    t_end = time.perf_counter_ns()
    _PERF_ACC_MICROS[name] += (t_end - t_start) // 1000
    _PERF_COUNTS[name] += 1


def clear_perf_stats():
    _PERF_ACC_MICROS.clear()
    _PERF_COUNTS.clear()


def print_perf_stats():
    if not _PERF_TIMING_ENABLED:
        return
    ms = _PERF_ACC_MICROS
    ns = _PERF_COUNTS
    width = max(len(k) for k in ms)
    print(f"{'method'.ljust(width)} {'total_sec':>11} {'count':>10} {'ops/s':>10}")
    for k in _PERF_ACC_MICROS:
        print(
            f"{k.ljust(width)} {ms[k]/1e6:>10.3f}s {ns[k]: 10} {ns[k]/(ms[k]/1e6):>10.1f}"
        )


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


class PurePyBoard:
    """Numpy representation of a hexz board.

    A board is represented by an (N, 10, 11) numpy array. Each 10x11 channel is
    a one-hot encoding of the presence of specific type of piece/obstacle/etc.
    The channels are:

    * 0: flags by P1
    * 1: cell value 1-5 for P1
    * 2: cells blocked for P1 (any occupied cell or a cell next to a 5)
    * 3: next value for P1
    * 4: flags by P2
    * 5: cell value 1-5 for P2
    * 6: cells blocked for P2
    * 7: next value for P2
    * 8: grass cells with value 1-5

    An action is specified by a (2, 10, 11) numpy array. The first 10x11 channel
    represents a flag move, the second one represents a regular cell move. A
    flag move must have a single 1 set, a normal move must have a single value
    1-5 set.
    """

    # Used to quickly get the indices of neighbor cells.
    neighbors = _init_neighbors_map()

    def __init__(self, other=None, dtype=np.float32, board=None):
        """Generates a new randomly initialized board or returns a copy of other, if set."""
        if other is not None:
            self.b = other.b.copy()
            self.nflags = list(other.nflags)
            return
        if board is not None:
            self.b = board
            self.nflags = [3 - int(board[0].sum()), 3 - int(board[4].sum())]
            return
        self.b = np.zeros((9, 11, 10), dtype=dtype)
        self.nflags = [3, 3]  # number of flags remaining per player
        # Even rows have 10 cells, odd rows only 9, so mark the last cell in odd rows as blocked for P1+P2.
        self.b[2, [1, 3, 5, 7, 9], 9] = 1
        self.b[6, [1, 3, 5, 7, 9], 9] = 1
        # 2-tuple of valid indices in each slice.
        free_cells = (1 - self.b[2]).nonzero()
        # 15 randomly placed stones.
        rng = np.random.default_rng()
        stones = rng.choice(np.arange(0, len(free_cells[0])), replace=False, size=15)
        self.b[2, free_cells[0][stones], free_cells[1][stones]] = 1
        self.b[6, free_cells[0][stones], free_cells[1][stones]] = 1
        free_cells = (1 - self.b[2]).nonzero()
        # 5 grass cells
        grass = rng.choice(np.arange(0, len(free_cells[0])), replace=False, size=5)
        self.b[8, free_cells[0][grass], free_cells[1][grass]] = [1, 2, 3, 4, 5]
        self.b[2, free_cells[0][grass], free_cells[1][grass]] = 1
        self.b[6, free_cells[0][grass], free_cells[1][grass]] = 1

    @classmethod
    def from_numpy(cls, board):
        if board.shape != (9, 11, 10):
            raise ValueError("Invalid board shape: " + str(board.shape))
        return cls(board=board)

    def b_for(self, player):
        """Returns the underlying ndarray representing the board, oriented for the given player.
        Only returns a copy if the board is not already oriented for the given player.
        """
        if player == 0:
            return self.b
        b = self.b.copy()
        b[0:4], b[4:8] = self.b[4:8], self.b[0:4]
        return b

    def quickview(self):
        """Returns a single slice of the board with different cell types encoded as -/+ numbers."""
        return (self.b[0] * 8) + self.b[1] - (self.b[4] * 8) - self.b[5]

    def score(self):
        """Returns the current score as a 2-tuple."""
        return (self.b[1].sum(), self.b[5].sum())

    def result(self):
        """Returns the final result of the board.

        1 (player 0 won), 0 (draw), -1 (player 1 won).
        """
        p0, p1 = self.score()
        return np.sign(p0 - p1)

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
        b[typ + player * 4, r, c] = val
        played_flag = typ == 0
        # Block played cell for both players.
        b[2, r, c] = 1
        b[6, r, c] = 1
        # Set next value to 0 for occupied cell.
        b[3, r, c] = 0
        b[7, r, c] = 0
        # Block neighboring cells if a 5 was played.
        nx, ny = Board.neighbors[(r, c)]
        # Update next value of neighboring cells. If we played a flag, the next value is 1.
        if played_flag:
            next_val = 1
            self.nflags[player] -= 1
        else:
            next_val = val + 1
        if next_val <= 5:
            for nr, nc in zip(nx, ny):
                if b[2 + player * 4, nr, nc] == 0:
                    # Cell is not blocked yet.
                    if next_val > 5:
                        b[3 + player * 4, nr, nc] = 0
                    if b[3 + player * 4, nr, nc] == 0:
                        b[3 + player * 4, nr, nc] = next_val
                    elif b[3 + player * 4, nr, nc] > next_val:
                        b[3 + player * 4, nr, nc] = next_val
        else:
            # Played a 5: block neighboring cells and clear next value.
            b[2 + player * 4, nx, ny] = 1
            b[3 + player * 4, nx, ny] = 0  # Clear next value.

        # Occupy neighboring grass cells.
        if not played_flag:
            self.occupy_grass(player, r, c)

    def occupy_grass(self, player, r, c):
        """Occupies the neighboring grass cells of move_idx (a 3-tuple index into a move) for player.

        Expects that the move has already been played.
        """
        nx, ny = Board.neighbors[(r, c)]
        for i, j in zip(nx, ny):
            grass_val = self.b[8, i, j]
            if grass_val > 0 and grass_val <= self.b[1 + player * 4, r, c]:
                # Occupy: first remove grass
                self.b[8, i, j] = 0
                # the rest is exactly like playing a move.
                self.make_move(player, (1, r, c, grass_val))

    @timing
    def next_moves(self, player):
        """Returns all possible next moves.

        A move is represented as a (2, 11, 10) ndarray. The first slice represents
        flag moves, the second one represents normal moves. A flag move will have exactly
        one element set to 1 in slice 0. A normal move will have exactly one element set to
        1-5 in slice 1.
        """
        moves = []
        # Do we have unoccupied cells and flags left? Then we can place another one.
        if self.nflags[player] > 0:
            # Flag in any unoccupied cell.
            rs, cs = np.nonzero(
                self.b[2 + player * 4] == 0
            )  # funky way to get indices for all free cells.
            moves.extend((0, r, c, 1) for r, c in zip(rs, cs))
        # Collect all cells with a non-zero next value.
        rs, cs = np.nonzero(self.b[3 + player * 4])
        moves.extend((1, r, c, self.b[3 + player * 4, r, c]) for r, c in zip(rs, cs))
        return moves

    def validate(self):
        """Checks that this instance represents a valid board.

        Raises an exeption if that is not the case.
        """
        b = self.b
        if np.any(b[2, [1, 3, 5, 7, 9], 9] == 0) or np.any(b[6, [1, 3, 5, 7, 9], 9] == 0) :
            raise ValueError("Invalid cells are not marked as blocked")
        if np.any(b[1] * b[5]):
            raise ValueError("Both players have a value in the same cell.")
        if np.any(b[0] * b[4]):
            raise ValueError("Both players have a flag in the same cell.")
        if np.any(b[2] * (b[0] + b[1]) != (b[0] + b[1])):
            raise ValueError("Not all occupied cells are marked as blocked for P1.")
        if np.any(b[6] * (b[4] + b[5]) != (b[4] + b[5])):
            raise ValueError("Not all occupied cells are marked as blocked for P2.")
        if np.any(b[0] * b[1]):
            raise ValueError("P1 has a value in a cell with a flag.")
        if np.any(b[4] * b[5]):
            raise ValueError("P2 has a value in a cell with a flag.")
        if np.any(b[3] * (b[0] + b[1] + b[2])):
            raise ValueError("P1 has a next value in an occupied cell.")
        if np.any(b[7] * (b[4] + b[5] + b[6])):
            raise ValueError("P2 has a next value in a blocked cell.")
        grass_values, grass_counts = np.unique(b[8], return_counts=True)
        if not (set(grass_values) <= set([0, 1, 2, 3, 4, 5])):
            raise ValueError(f"Wrong values for grass cells: {grass_values}")
        if len(grass_counts) > 1 and grass_counts[1:].max() > 1:
            raise ValueError(f"Duplicate grass cells: {grass_values}, {grass_counts}")
        if b.max() > 5:
            raise ValueError("Max value is > 5.")
        if b.min() < 0:
            raise ValueError("Min value is < 0.")


class CBoard(PurePyBoard):
    """Like a PurePyBoard, but uses the fast C implementations where available.

    A CBoard currently only works with np.float32 dtype.
    """

    @timing
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=np.float32, **kwargs)

    @timing
    def make_move(self, player, move):
        hexc.c_make_move(self, player, move)

    @timing
    def occupy_grass(self, player, r, c):
        hexc.c_occupy_grass(self, player, r, c)


# Use the C implementation.
Board = CBoard


class Node:
    """Nodes of the MCTS tree."""

    def __init__(self, parent, player, move):
        self.parent = parent
        self.player = player
        self.move = move
        self.wins = 0.0
        self.visit_count = 0
        self.children = []

    def uct(self):
        if self.parent.visit_count == 0:
            return 0.5
        if self.visit_count == 0:
            win_rate = 0.5
            adjusted_count = 1
        else:
            win_rate = self.wins / self.visit_count
            adjusted_count = self.visit_count
        return win_rate + 1.0 * np.sqrt(
            np.log(self.parent.visit_count) / adjusted_count
        )

    def __str__(self):
        return f"Node(p={self.player}, m={self.move}, w={self.wins/self.visit_count:.3f}, n={self.visit_count}, u={self.uct():.3f}, cs={len(self.children)})"

    def __repr__(self):
        return str(self)

    def move_likelihoods(self):
        """Returns the move likelihoods for all children as a (2, 11, 10) ndarray.

        The ndarray indicates the likelihoods (based on visit count) for flags
        and normal moves. It sums to 1.
        """
        p = np.zeros((2, 11, 10))
        for child in self.children:
            typ, r, c, _ = child.move
            p[typ, r, c] = child.visit_count
        return p / p.sum()

    def best_child(self):
        """Returns the best among all children.

        The best child is the one with the greatest visit count, a common
        choice in the MCTS literature.
        """
        return max(self.children, default=None, key=lambda n: n.visit_count)


class Example:
    """Data holder for one step of a fully played MCTS game."""

    def __init__(self, game_id, board, move_probs, turn, result):
        """
        Args:
            game_id: arbitrary string identifying the game that this example belongs to.
            board: the board (Board.b) as an (N, 11, 10) ndarray.
            move_probs: (2, 11, 10) ndarray of move likelihoods.
            turn: 0 or 1, indicating which player's turn it was.
            result: -1, 0, 1, indicating the player that won (1 = player 0 won).
        """
        self.game_id = game_id
        self.board = board
        self.move_probs = move_probs
        self.turn = turn
        self.result = result

    @classmethod
    def save_all(self, path, examples, mode="a", dtype=np.float32):
        """Saves the examples in a HDF5 file at the given path.

        Args:
            path: a file path, e.g. "/path/to/examples.h5"
            examples: a list of Example instances.
            mode: 'a' (the default) to append or create, 'w' to truncate or create.
        """
        with h5py.File(path, mode) as h5:
            offset = len(h5)
            for i, ex in enumerate(examples):
                grp = h5.create_group(f"{i+offset:08}")
                grp.attrs["game_id"] = ex.game_id
                grp.create_dataset("board", data=ex.board)
                grp.create_dataset("move_probs", data=ex.move_probs)
                grp.create_dataset("turn", data=np.array([ex.turn], dtype=dtype))
                grp.create_dataset("result", data=np.array([ex.result], dtype=dtype))

    @classmethod
    def load_all(cls, path):
        """This method is for testing only.

        Use a HDF5Dataset to access the examples from PyTorch."""
        examples = []
        with h5py.File(path, "r") as h5:
            for grp in h5:
                ex = h5[grp]
                turn = ex["turn"][0]
                result = ex["result"][0]
                examples.append(
                    Example(
                        ex.attrs["game_id"],
                        ex["board"][:],
                        ex["move_probs"][:],
                        turn,
                        result,
                    )
                )
        return examples


class MCTS:
    """Monte Carlo tree search."""

    def __init__(self, board, game_id=None):
        self.board = board
        # In the root node it's player 1's "fake" turn.
        # This has the desired effect that the root's children will play
        # as player 0, who makes the first move.
        self.root = Node(None, 1, None)
        self.rng = np.random.default_rng()
        if not game_id:
            game_id = uuid4().hex[:12]
        self.game_id = game_id

    def rollout(self, board, player):
        """Play a random game till the end, starting with board and player on the move."""
        for i in range(200):
            moves = board.next_moves(player)
            if not moves:
                # No more moves for the player. See if the other player can continue.
                player = 1 - player
                moves = board.next_moves(player)
            if not moves:
                return board.result()
            b.make_move(player, moves[self.rng.integers(0, len(moves))])
            player = 1 - player
        raise ValueError("Never finished rollout")

    def backpropagate(self, node, result):
        while node:
            node.visit_count += 1
            if node.player == 0:
                node.wins += (result + 1) / 2
            else:
                node.wins += (-result + 1) / 2
            node = node.parent

    def size(self):
        q = [self.root]
        s = 0
        while q:
            n = q.pop()
            s += 1
            q.extend(n.children)
        return s

    def run(self):
        """Runs a single round of the MCTS loop."""
        b = Board(self.board)
        n = self.root
        # Find leaf node.
        while n.children:
            best = None
            best_uct = -1
            for c in n.children:
                c_uct = c.uct()
                if c_uct > best_uct:
                    best = c
                    best_uct = c_uct
            b.make_move(best.player, best.move)
            n = best
        # Reached a leaf node: expand
        player = 1 - n.player  # Usually it's the other player's turn.
        moves = b.next_moves(player)
        if not moves:
            # No more moves for player. Try other player.
            player = 1 - player
            moves = b.next_moves(player)
        if not moves:
            # Game is over
            self.backpropagate(n, b.result())
            return
        # Rollout time!
        for move in moves:
            n.children.append(Node(n, player, move))
        c = n.children[self.rng.integers(0, len(n.children))]
        b.make_move(c.player, c.move)
        result = self.rollout(b, 1 - c.player)
        self.backpropagate(c, result)

    def play_game(self, runs_per_move=500):
        """Plays one full game and returns the move likelihoods per move and the final result.

        Args:
            runs_per_move: number of MCTS runs to make per move.
        """
        examples = []
        result = None
        n = 0
        started = time.perf_counter()
        while n < 200:
            for i in range(runs_per_move):
                self.run()
            best_child = self.root.best_child()
            if not best_child:
                # Game over
                result = self.board.result()
                break
            examples.append(
                Example(
                    self.game_id,
                    self.board.b.copy(),
                    self.root.move_likelihoods(),
                    best_child.player,
                    None,
                )
            )
            # Make the move.
            self.board.make_move(best_child.player, best_child.move)
            self.root = best_child
            self.root.parent = None  # Allow GC and avoid backprop further up.
            if n < 5 or n % 10 == 0:
                print(
                    f"Iteration {n}: visit_count:{best_child.visit_count} ",
                    f"move:{best_child.move} player:{best_child.player} score:{self.board.score()}",
                )
            n += 1
        if n == 200:
            raise ValueError(f"Iterated {n} times. Something's fishy.")
        elapsed = time.perf_counter() - started
        print(
            f"Done in {elapsed:.3f}s after {n} moves. Final score: {self.board.score()}."
        )
        # Update all examples with result.
        for ex in examples:
            ex.result = result
        return examples


class HDF5Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset implementation to read Hexz samples from HDF5.
    Supports reading from multiple files.
    """

    def __init__(self, path, in_mem=False, lazy_init=True):
        """Builds a new dataset that reads from the .h5 file(s) pointed to by path.

        Args:
            path: can be a list of paths, a glob, or the path of a single HDF5 .h5 file.
            in_mem: if True, *all* HDF5 datasets will be pre-loaded into memory.
                This can speed up training significantly, but ofc only works for sufficiently
                small datasets.
        """
        if isinstance(path, list):
            self.paths = [p for p in path if os.path.isfile(p)]
        else:
            self.paths = glob.glob(path)
        if not self.paths:
            raise ValueError(f"No files found at {path}")
        self.in_mem = in_mem
        if not lazy_init:
            self.init()
        else:
            # We need to know the size of our dataset in the master process, at it generates
            # the batch indices. Since we cannot pickle h5py objects, but pickling is required
            # on macos in a multiprocessing context (since spawn() is used to create the subprocesses),
            # we precompute the size by looking at all datasets once.
            self.size = sum(len(h5py.File(p, "r")) for p in self.paths)

    @timing
    def init(self):
        """Must be called to initialize this dataset.

        This effectively implements a lazy initialization, allowing us to use an HDF5Dataset
        in worker subprocesses. If we would initialize in __init__ already, we would have to
        pickle h5py objects, which is not supported.
        """
        self.h5s = []
        size = 0
        for p in self.paths:
            h = h5py.File(p, "r")
            l = len(h)
            if l > 0:
                self.h5s.append(h)
                size += l
        if not self.h5s:
            raise FileNotFoundError(f"No examples found in {self.paths}")
        self.size = size
        self.memcache = None
        self.end_idxs = list(np.array([len(h) for h in self.h5s]).cumsum() - 1)
        self.keys = [list(h.keys()) for h in self.h5s]
        if self.in_mem:
            self.init_memcache()

    def init_memcache(self):
        """Loads all HDF5 data into self.memcache. Expects that this object is otherwise fully initialized."""
        mc = [None] * len(self)
        for i in range(len(mc)):
            mc[i] = self[i]
        self.memcache = mc

    def __getitem__(self, k):
        """Returns the board as the X and two labels: move likelihoods (2, 11, 10) and (1) result.

        The collation function the PyTorch DataLoader handles this properly and returns a tuple
        of batches for the labels.
        """
        if self.memcache:
            return self.memcache[k]
        i = bisect_left(self.end_idxs, k)
        h5 = self.h5s[i]
        if i > 0:
            k -= self.end_idxs[i - 1] + 1
        data = h5[self.keys[i][k]]
        # data["turn"] ignored for now.
        return data["board"][:], (data["move_probs"][:], data["result"][:])

    def __len__(self):
        return self.size


class HexzNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        board_channels = 9
        cnn_filters = 128
        num_cnn_blocks = 5
        self.cnn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        board_channels,
                        cnn_filters,
                        kernel_size=3,
                        bias=False,
                        padding="same",
                    ),  # [N, cnn_channels, 11, 10]
                    nn.BatchNorm2d(cnn_filters),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    nn.Conv2d(
                        cnn_filters,
                        cnn_filters,
                        kernel_size=3,
                        bias=False,
                        padding="same",
                    ),  # [cnn_channels, cnn_channels, 11, 10]
                    nn.BatchNorm2d(cnn_filters),
                    nn.ReLU(),
                )
                for i in range(num_cnn_blocks - 1)
            ]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(cnn_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 11 * 10, 2 * 11 * 10),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(cnn_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(11 * 10, 11 * 10),
            nn.ReLU(),
            nn.Linear(11 * 10, 1),
        )

    def forward(self, x):
        for b in self.cnn_blocks:
            x = b(x)
        pi = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class NNode:
    """Nodes of the NeuralMCTS tree."""

    def __init__(self, parent: 'Optional[NNode]', player: int, move):
        self.parent = parent
        self.player = player
        self.move = move
        self.wins = 0.0
        self.visit_count: int = 0
        self.children: list[NNode] = []
        self.move_probs = None

    @timing
    def uct(self):
        typ, r, c, _ = self.move
        pvc = self.parent.visit_count
        vc = self.visit_count
        if vc == 0:
            q = 0.5
            vc = 1
        else:
            q = self.wins / vc
        return q + self.parent.move_probs[typ, r, c] * np.sqrt(np.log(pvc) / vc)

    def puct(self):
        typ, r, c, _ = self.move
        pr = self.parent.move_probs
        if self.visit_count == 0:
            q = 0.0
        else:
            q = self.wins / self.visit_count
        return q + pr[typ, r, c] * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)

    def move_likelihoods(self, dtype=np.float32):
        """Returns the move likelihoods for all children as a (2, 11, 10) ndarray.

        The ndarray indicates the likelihoods (based on visit count) for flags
        and normal moves. It sums to 1.
        """
        p = np.zeros((2, 11, 10), dtype=dtype)
        for child in self.children:
            typ, r, c, _ = child.move
            p[typ, r, c] = child.visit_count
        return p / p.sum()

    def best_child(self) -> 'NNode':
        """Returns the best among all children.

        The best child is the one with the greatest visit count, a common
        choice in the MCTS literature.
        """
        return max(self.children, default=None, key=lambda n: n.visit_count)


class NeuralMCTS:
    """Monte Carlo tree search with a neural network (AlphaZero style)."""

    @timing
    def __init__(
        self,
        board: Board,
        model: HexzNeuralNetwork,
        game_id: Optional[str] = None,
        turn: int = 0,
        device: str = "cpu",
        dtype = torch.float32,
    ):
        self.board = board
        self.model = model
        self.rng = np.random.default_rng()
        if not game_id:
            game_id = uuid4().hex[:12]
        self.game_id = game_id
        self.device = device
        self.dtype = dtype

        model.eval()  # Training is done outside of NeuralMCTS
        # The root node has the opposite player whose actual turn it is.
        # This has the desired effect that the root's children will play
        # as the other player, who makes the first move.
        self.root: NNode = NNode(None, 1 - turn, None)
        # Predict move probabilities for root and board value estimate up front.
        self.root.move_probs, self.value = self.predict(board, turn)

    def backpropagate(self, node, result):
        while node:
            node.visit_count += 1
            if node.player == 0:
                node.wins += (result + 1) / 2
            else:
                node.wins += (-result + 1) / 2
            node = node.parent

    def size(self):
        q = [self.root]
        s = 0
        while q:
            n = q.pop()
            s += 1
            q.extend(n.children)
        return s

    @timing
    @torch.inference_mode()
    def predict(self, board, player):
        """Predicts move probabilities and value for the given board and player."""
        b = board.b_for(player)
        X = torch.from_numpy(b).to(self.device, dtype=self.dtype)
        pred_pr, pred_val = self.model(torch.unsqueeze(X, 0))
        pred_pr = torch.exp(pred_pr).reshape((2, 11, 10))
        return pred_pr.numpy(force=self.device != "cpu"), pred_val.item()

    def find_leaf(self, b):
        n = self.root
        while n.children:
            best = None
            best_uct = -1
            for c in n.children:
                c_uct = c.puct()
                if c_uct > best_uct:
                    best = c
                    best_uct = c_uct
            b.make_move(best.player, best.move)
            n = best
        return n

    @timing
    def run(self):
        """Runs a single round of the neural MCTS loop."""
        b = Board(self.board)
        n = self.root
        # Find leaf node.
        with timing_ctx("run_find_leaf"):
            n = hexc.c_find_leaf(b, n)
            # n = self.find_leaf(b)
        # Reached a leaf node: expand
        player = 1 - n.player  # Usually it's the other player's turn.
        moves = b.next_moves(player)
        if not moves and n is not self.root:
            # No more moves for player. Try other player, but not for the first move.
            player = 1 - player
            moves = b.next_moves(player)
        if not moves:
            # Game is over
            self.backpropagate(n, b.result())
            return
        for move in moves:
            n.children.append(NNode(n, player, move))
        # Neural network time! Predict value and policy for next move.
        n.move_probs, value = self.predict(b, player)
        self.backpropagate(n, value)

    def play_game(self, runs_per_move=500, max_moves=200, progress_queue=None):
        """Plays one full game and returns the move likelihoods per move and the final result.

        Args:
            runs_per_move: number of MCTS runs to make per move.
        """
        examples = []
        result = None
        n = 0
        started = time.perf_counter()
        while n < max_moves:
            for i in range(runs_per_move):
                self.run()
            best_child = self.root.best_child()
            if not best_child:
                # Game over
                result = self.board.result()
                break
            # Record example. Examples always contain the board as seen by the player whose turn it is.
            examples.append(
                Example(
                    self.game_id,
                    self.board.b_for(best_child.player).copy(),
                    self.root.move_likelihoods(),
                    best_child.player,
                    None,
                )
            )
            if progress_queue:
                progress_queue.put({
                    'examples': 1,
                })
            # Make the move.
            self.board.make_move(best_child.player, best_child.move)
            self.root = best_child
            self.root.parent = None  # Allow GC and avoid backprop further up.
            if n < 5 or n % 10 == 0:
                print(
                    f"Iteration {n} @{time.perf_counter() - started:.3f}s: visit_count:{best_child.visit_count} ",
                    f"move:{best_child.move} player:{best_child.player} score:{self.board.score()}",
                )
            n += 1
        elapsed = time.perf_counter() - started
        if n == max_moves:
            print(
                f"Reached max iterations ({n}) after {elapsed:.3f}s. Returning early."
            )
            return []
        print(
            f"Done in {elapsed:.3f}s after {n} moves. Final score: {self.board.score()}."
        )
        # Update all examples with result.
        for ex in examples:
            ex.result = result
        return examples


def load_model(path, map_location='cpu'):
    model = HexzNeuralNetwork()
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model


def time_gameplay(device):
    clear_perf_stats()
    b = Board()
    model = HexzNeuralNetwork().to(device)
    m = NeuralMCTS(b, model, device=device)
    _ = m.play_game(800, max_moves=200)
    print_perf_stats()


def record_examples(progress_queue: mp.Queue, args):
    worker_id = os.getpid()
    print(f"Worker {worker_id} started.")
    # disable_perf_stats()
    started = time.time()
    num_games = 0
    model_path = path_to_latest_model(args.model)
    device = args.device
    model = load_model(model_path).to(device)
    # model = torch.compile(model)
    print(f"W{worker_id}: loaded model from {model_path}")
    examples_file = os.path.join(
        args.output_dir, f"examples-{time.strftime('%Y%m%d-%H%M%S')}-{worker_id}.h5"
    )
    print(f"W{worker_id}: Appending game examples to {examples_file}. {args.runs_per_move=}")
    while time.time() - started < args.max_seconds and num_games < args.max_games:
        b = Board()
        m = NeuralMCTS(b, model, device=device)
        examples = m.play_game(runs_per_move=args.runs_per_move, progress_queue=progress_queue)
        Example.save_all(examples_file, examples)
        num_games += 1
        progress_queue.put({
            'games': 1,
            'done': False,
        })
    progress_queue.put({
        'done': True,
    })
    print_perf_stats()


def record_examples_mp(args):
    num_workers = args.num_workers
    progress_queue = mp.Queue()
    procs = []
    started = time.perf_counter()
    for _ in range(num_workers):
        p = mp.Process(target=record_examples, args=(progress_queue, args))
        p.start()
        procs.append(p)
    # Process status updates until all workers are done
    running = num_workers
    num_examples = 0
    num_games = 0
    while running > 0:
        msg = progress_queue.get()
        if msg.get('done', False):
            running -= 1
        num_examples += msg.get('examples', 0)
        num_games += msg.get('games', 0)
        elapsed = time.perf_counter()-started
        print(f"At {elapsed:.1f}s: examples:{num_examples} games:{num_games} ({num_examples/elapsed:.1f} examples/s).")
    for p in procs:
        p.join()


def parse_args():
    parser = argparse.ArgumentParser(description="Hexz NeuralMCTS")
    parser.add_argument(
        "--mode",
        type=str,
        default="selfplay",
        help="Mode to execute: selfplay to generate examples, generate to generate a new randomly initialized model, train to ... train",
        choices=("selfplay", "train", "generate", "print"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device to use",
        choices=("cpu", "cuda", "mps"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        help="Path to the model to use. If this is a directory, picks the latest model checkpoint.",
    )
    parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Path or glob to the examples to use during training. Default: '{--model}/examples/*.h5'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory into which outputs are written.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=1000,
        help="Maximum number of games to play during self-play.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=60,
        help="Maximum number of seconds to play during self-play.",
    )
    parser.add_argument(
        "--runs-per-move", type=int, default=800, help="Number of MCTS runs per move."
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size to use during training."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to shuffle the examples during training.",
    )
    parser.add_argument(
        "--in-mem",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to load all HDF5 example data into memory for training.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Determines the pin_memory value used in the PyTorch DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Determines the num_workers value used in the PyTorch DataLoader. Set to 0 to disable multi-processing.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, existing outputs may be overriden.",
    )
    return parser.parse_args()


def path_to_latest_model(base_path):
    m = base_path
    if os.path.isfile(m):
        return m
    if os.path.isdir(m):
        # Expected dir structure: {model_name}/checkpoints/{checkpoint_number}/model.pt
        if os.path.isfile(os.path.join(m, "model.pt")):
            return os.path.join(m, "model.pt")
        if os.path.basename(m) == "checkpoints":
            models = glob.glob(os.path.join(m, "*", "model.pt"))
        else:
            models = glob.glob(os.path.join(m, "checkpoints", "*", "model.pt"))
        if not models:
            raise FileNotFoundError(f"No model checkpoints found in {m}.")
        try:
            models = sorted(models, key=lambda m: int(m.rsplit("/", 2)[-2]))
            return models[-1]
        except ValueError:
            raise ValueError(f"Non-numeric checkpoint dirs: {models}")
    else:
        raise FileNotFoundError(
            f"Model path {base_path} is neither a file nor a directory."
        )


def worker_init_fn(worker_id):
    """Called to initialize a DataLoader worker if multiprocessing is used."""
    w = torch.utils.data.get_worker_info()
    if w:
        if isinstance(w.dataset, HDF5Dataset):
            w.dataset.init()
    else:
        print("worker_init_fn: Called in main process!")


def train_model(args):
    num_workers = args.num_workers
    batch_size = args.batch_size
    num_epochs = args.epochs
    model_path = path_to_latest_model(args.model)
    examples_path = args.examples
    if not examples_path:
        examples_path = os.path.join(os.path.dirname(model_path), "examples/*.h5")
    device = args.device
    model = load_model(model_path).to(device)
    started = time.perf_counter()
    with print_time("HDF5Dataset()"):
        ds = HDF5Dataset(examples_path, in_mem=args.in_mem, lazy_init=num_workers > 0)
    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    # loader.num_workers > 0 uses separate processes to load data.
    # pin_memory=True should speed up the CPU->GPU data transfer
    #
    # https://github.com/pytorch/pytorch/issues/77799
    # Discussion about mps vs cpu device performance.

    loader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    print(f"DataLoader.num_workers: {loader.num_workers}")
    print(f"torch.autograd.gradcheck:  {torch.autograd.gradcheck}")
    for X, (y_pr, y_val) in loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}: {X.dtype}")
        print(f"Shape of y_pr [N, C, H, W]: {y_pr.shape}: {y_pr.dtype}")
        print(f"Shape of y_val [N, V]: {y_val.shape}: {y_val.dtype}")
        break

    pr_loss_fn = nn.CrossEntropyLoss()
    val_loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # def trace_handler(prof):
    #     print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

    for epoch in range(num_epochs):
        epoch_started = time.perf_counter()
        examples_processed = 0
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
        #                             schedule=torch.profiler.schedule(
        #                                 wait=5,
        #                                 warmup=5,
        #                                 active=1,
        #                                 repeat=1,
        #                             ),
        #                             on_trace_ready=trace_handler) as prof:
        # with torch.profiler.record_function("batch_iteration"):
        for batch, (X, (y_pr, y_val)) in enumerate(loader):
            # Send to device.
            X = X.to(device)
            y_pr = y_pr.to(device)
            y_val = y_val.to(device)

            # Predict
            pred_pr, pred_val = model(X)

            # Compute loss
            pr_loss = pr_loss_fn(pred_pr.flatten(1), y_pr.flatten(1))
            val_loss = val_loss_fn(pred_val, y_val)
            loss = pr_loss + val_loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            examples_processed += len(X)
            # prof.step()
            if batch % 100 == 0:
                now = time.perf_counter()
                print(
                    f"Examples/s in epoch #{epoch}: {examples_processed/(now-epoch_started):.1f}/s ({examples_processed} total)"
                )
                print(
                    f"pr_loss: {pr_loss.item():.6f}, val_loss: {val_loss.item():.6f} @{epoch}/{examples_processed}"
                )
    output_path = "/tmp/model.pt"  # TODO: make configurable
    torch.save(model.state_dict(), output_path)
    total_duration = time.perf_counter() - started
    print(f"Done after {total_duration:.1f}s. Saved new model to {output_path}.")


def generate_model(args):
    model = HexzNeuralNetwork()
    path = args.model
    if os.path.isdir(path):
        path = os.path.join(path, "model.pt")
    if not args.force and os.path.isfile(path):
        print(f"generate_model: error: a model already exists at {path}.")
        return
    torch.save(model.state_dict(), path)
    print(
        f"Generated randomly initialized model with {sum(p.numel() for p in model.parameters())} parameters at {path}."
    )


def print_model_info(args):
    model_path = path_to_latest_model(args.model)
    model = load_model(model_path)
    print(model)


def main():
    args = parse_args()
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"torch version: {torch.__version__}")
    print(f"numpy version: {np.__version__}")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Device cuda not available, falling back to cpu.")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("Device mps not available, falling back to cpu.")
        args.device = "cpu"
    print("Using device:", args.device)
    # model_path = "../../hexz-models/models/flagz/genesis/generations/0/model.pt"
    # One-off: Save an initial model.
    # model = HexzNeuralNetwork()
    # save_model(model, os.path.join(
    #     os.path.dirname(sys.argv[0]),
    #     model_path)
    # )
    if args.mode == "selfplay":
        record_examples_mp(args)
    elif args.mode == "train":
        train_model(args)
    elif args.mode == "generate":
        generate_model(args)
    elif args.mode == "print":
        print_model_info(args)


if __name__ == "__main__":
    main()
