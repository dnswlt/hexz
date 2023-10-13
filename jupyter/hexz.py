import argparse
import collections
from contextlib import contextmanager
from functools import wraps
import glob
import h5py
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
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
        print(f"{k.ljust(width)} {ms[k]/1e6:>10.3f}s {ns[k]: 10} {ns[k]/(ms[k]/1e6):>10.1f}")
    

def _init_neighbors_map():
    """Returns a dict mapping all valid (r, c) indices to their neighbor indices.
    
    The neighbor indices are represented as (row, column) tuples."""
    def _valid_idx(r_c):
        r, c = r_c
        return r >= 0 and r < 11 and c >= 0 and c < 10-r%2
    
    result = {}
    for r in range(11):
        shift = r%2 # Depending on the row, neighbors below and above are shifted.
        for c in range(10-r%2):
            ns = filter(_valid_idx, [
                (r, c+1),
                (r-1, c+shift),
                (r-1, c-1+shift),
                (r, c-1),
                (r+1, c-1+shift),
                (r+1, c+shift),
            ])
            nr, nc = zip(*ns)  # unzip
            result[(r, c)] = (np.array(nr), np.array(nc))
            
    return result


class PurePyBoard:
    """Numpy representation of a hexz board."""

    # Used to quickly get the indices of neighbor cells.
    neighbors = _init_neighbors_map()
    
    def __init__(self, other=None, dtype=np.float32):
        """Generates a new randomly initialized board or returns a copy of other, if set."""
        if other:
            self.b = other.b.copy()
            self.nflags = list(other.nflags)
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
        b[typ + player*4, r, c] = val
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
                if b[2 + player*4, nr, nc] == 0:
                    if next_val > 5:
                        b[3 + player*4, nr, nc] = 0
                    if b[3 + player*4, nr, nc] == 0:
                        b[3 + player*4, nr, nc] = next_val
                    elif b[3 + player*4, nr, nc] > next_val:
                        b[3 + player*4, nr, nc] = next_val
        else:
            # Played a 5: block neighboring cells and clear next value.
            b[2 + player*4, nx, ny] = 1
            b[3 + player*4, nx, ny] = 0  # Clear next value.

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
            if grass_val > 0 and grass_val <= self.b[1 + player*4, r, c]:
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
            rs, cs = np.nonzero(self.b[2 + player*4] == 0)  # funky way to get indices for all free cells.
            moves.extend((0, r, c, 1) for r, c in zip(rs, cs))
        # Collect all cells with a non-zero next value.
        rs, cs = np.nonzero(self.b[3 + player*4])
        moves.extend((1, r, c, self.b[3 + player*4, r, c]) for r, c in zip(rs, cs))
        return moves

    
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
        return (
            win_rate + 
            1.0 * np.sqrt(np.log(self.parent.visit_count) / adjusted_count)
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
    def save_all(self, path, examples, mode="a"):
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
                grp.create_dataset("turn", data=np.array([ex.turn]))
                grp.create_dataset("result", data=np.array([ex.result]))
    
    @classmethod
    def load_all(cls, path):
        """This method is for testing only.
        
        Use a HexzDataset to access the examples from PyTorch."""
        examples = []
        with h5py.File(path, "r") as h5:
            for grp in h5:
                ex = h5[grp]
                turn = ex["turn"][0]
                result = ex["result"][0]
                examples.append(
                    Example(ex.attrs["game_id"], ex["board"][:], ex["move_probs"][:], turn, result)
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
            examples.append(Example(self.game_id, self.board.b.copy(), 
                                    self.root.move_likelihoods(), best_child.player, None))
            # Make the move.
            self.board.make_move(best_child.player, best_child.move)
            self.root = best_child
            self.root.parent = None  # Allow GC and avoid backprop further up.
            if n < 5 or n%10 == 0:
                print(f"Iteration {n}: visit_count:{best_child.visit_count} ",
                      f"move:{best_child.move} player:{best_child.player} score:{self.board.score()}")
            n += 1
        if n == 200:
            raise ValueError(f"Iterated {n} times. Something's fishy.")
        elapsed = time.perf_counter() - started
        print(f"Done in {elapsed:.3f}s after {n} moves. Final score: {self.board.score()}.")
        # Update all examples with result.
        for ex in examples:
            ex.result = result
        return examples

class HexzDataset(torch.utils.data.Dataset):
    """PyTorch Dataset implementation to read Hexz samples from HDF5."""
    def __init__(self, path):
        self.h5 = h5py.File(path, "r")
        self.length = len(self.h5)
        self.keys = list(self.h5.keys())
        
    def __getitem__(self, k):
        """Returns the board as the X and two labels: move likelihoods (2, 11, 10) and (1) result.
        
        The collation function the PyTorch DataLoader handles this properly and returns a tuple
        of batches for the labels.
        """
        data = self.h5[self.keys[k]]
        # data["turn"][0] ignored for now.
        return data["board"][:], (data["move_probs"][:], data["result"][:])

    def __len__(self):
        return self.length
        

class HexzNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        board_channels = 9
        cnn_channels = 32
        self.features = nn.Sequential(
            # First CNN layer
            nn.Conv2d(board_channels, cnn_channels, kernel_size=3, bias=False),  # [N, cnn_channels, 9, 8]
            nn.BatchNorm2d(cnn_channels), 
            nn.ReLU(),
            # Second CNN layer
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, bias=False),  # # [N, cnn_channels, 7, 6]
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
        )
        fc_size = 256
        self.fc = nn.Sequential(
            nn.Linear(cnn_channels * 7 * 6, fc_size),
            nn.BatchNorm1d(fc_size), 
            nn.ReLU(),
            nn.Linear(fc_size, fc_size),
        )
        self.move_probs = nn.Linear(fc_size, 2 * 11 * 10)
        self.value = nn.Linear(fc_size, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        pi = self.move_probs(x)
        v = self.value(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class NNode:
    """Nodes of the NeuralMCTS tree."""
    def __init__(self, parent, player, move):
        self.parent = parent
        self.player = player
        self.move = move
        self.wins = 0.0
        self.visit_count = 0
        self.children = []
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
        return (
            q +
            self.parent.move_probs[typ, r, c] * np.sqrt(np.log(pvc) / vc)
        )

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

        
class NeuralMCTS:
    """Monte Carlo tree search with a neural network (AlphaZero style)."""
    def __init__(self, board, model, game_id=None, device='cpu', dtype=torch.float32):
        self.board = board
        self.model = model
        self.rng = np.random.default_rng()
        if not game_id:
            game_id = uuid4().hex[:12]
        self.game_id = game_id
        self.device = device
        self.dtype = dtype

        model.eval()  # Training is done outside of NeuralMCTS
        # In the root node it's player 1's "fake" turn. 
        # This has the desired effect that the root's children will play
        # as player 0, who makes the first move.
        self.root = NNode(None, 1, None)
        # Predict move probabilities for root up front.
        self.root.move_probs, _ = self.predict(board, 0)
        
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
        return pred_pr.numpy(force=self.device!='cpu'), pred_val.item()
    
    @timing
    def run(self):
        """Runs a single round of the neural MCTS loop."""
        b = Board(self.board)
        n = self.root
        # Find leaf node.
        with timing_ctx("run_find_leaf"):
            n = hexc.c_find_leaf(b, n)
            # while n.children:
            #     best = None
            #     best_uct = -1
            #     for c in n.children:
            #         c_uct = c.uct()
            #         if c_uct > best_uct:
            #             best = c
            #             best_uct = c_uct
            #     b.make_move(best.player, best.move)
            #     n = best
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
        for move in moves:
            n.children.append(NNode(n, player, move))
        # Neural network time! Predict value and policy for each child.
        n.move_probs, value = self.predict(b, player)
        self.backpropagate(n, value)
    
    def play_game(self, runs_per_move=500, max_moves=200):
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
            examples.append(Example(self.game_id, self.board.b_for(best_child.player).copy(),
                                    self.root.move_likelihoods(), best_child.player, None))
            # Make the move.
            self.board.make_move(best_child.player, best_child.move)
            self.root = best_child
            self.root.parent = None  # Allow GC and avoid backprop further up.
            if n < 5 or n%10 == 0:
                print(f"Iteration {n} @{time.perf_counter() - started:.3f}s: visit_count:{best_child.visit_count} ",
                        f"move:{best_child.move} player:{best_child.player} score:{self.board.score()}")
            n += 1
        elapsed = time.perf_counter() - started
        if n == max_moves:
            print(f"Reached max iterations ({n}) after {elapsed:.3f}s. Returning early.")
            return []
        print(f"Done in {elapsed:.3f}s after {n} moves. Final score: {self.board.score()}.")
        # Update all examples with result.
        for ex in examples:
            ex.result = result
        return examples


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = HexzNeuralNetwork()
    model.load_state_dict(torch.load(path))
    return model


def time_gameplay(device):
    clear_perf_stats()
    b = Board()
    model = HexzNeuralNetwork().to(device)
    m = NeuralMCTS(b, model, device=device)
    _ = m.play_game(800, max_moves=200)
    print_perf_stats()


def record_examples(args):
    disable_perf_stats()
    started = time.time()
    num_games = 0
    model_path = resolve_model_path(args)
    device = args.device
    model = load_model(model_path).to(device)
    print(f"Loaded model from {model_path}")
    examples_file = os.path.join(args.output_dir, f"examples-{time.strftime('%Y%m%d-%H%M%S')}.h5")
    print(f"Appending game examples to {examples_file}.")
    while time.time() - started < args.max_seconds and num_games < args.max_games:
        b = Board()
        m = NeuralMCTS(b, model, device=device)
        examples = m.play_game(runs_per_move=args.runs_per_move)
        Example.save_all(examples_file, examples)
        num_games += 1
        print(f"Played {num_games} games in {time.time() - started:.1f}s.")
    print("Done recording examples after {time.time() - started:.1f}s.")

def parse_args():
    parser = argparse.ArgumentParser(description="Hexz NeuralMCTS")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device to use", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--model", type=str, default=None, required=True,
                        help="Path to the model to use. If this is a directory, picks the latest model checkpoint.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory into which outputs are written.")
    parser.add_argument("--max-games", type=int, default=1000,
                        help="Maximum number of games to play during self-play.")
    parser.add_argument("--max-seconds", type=int, default=60,
                        help="Maximum number of seconds to play during self-play.")
    parser.add_argument("--runs-per-move", type=int, default=800,
                        help="Number of MCTS runs per move.")
    return parser.parse_args()


def resolve_model_path(args):
    m = args.model
    if os.path.isfile(m):
        return m
    if os.path.isdir(m):
        if os.path.isfile(os.path.join(m, "model.pt")):
            return os.path.join(m, "model.pt")
        if os.path.basename(m) == "checkpoints":
            models = glob.glob(os.path.join(m, "*", "model.pt"))
        else: 
            models = glob.glob(os.path.join(m, "checkpoints", "*", "model.pt"))
        if not models:
            raise ValueError(f"No model checkpoints found in {m}.")
        # Expected dir structure: {model_name}/checkpoints/{checkpoint_number}/model.pt
        return max(models, key=lambda p: int(p.rsplit("/", 2)[-2]))
    else:
        raise ValueError(f"Model path {args.model} is neither a file nor a directory.")
    

def main():
    args = parse_args()
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"torch version: {torch.__version__}")
    print(f"numpy version: {np.__version__}")
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Device cuda not available, falling back to cpu.")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("Device mps not available, falling back to cpu.")
        args.device = 'cpu'
    print("Using device:", args.device)
    # model_path = "../../hexz-models/models/flagz/genesis/generations/0/model.pt"
    # One-off: Save an initial model.
    # model = HexzNeuralNetwork()
    # save_model(model, os.path.join(
    #     os.path.dirname(sys.argv[0]),
    #     model_path)
    # )
    record_examples(args)


if __name__ == "__main__":
    main()
