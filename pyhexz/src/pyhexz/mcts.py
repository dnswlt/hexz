"""Vanilla MCTS for the Flagz game in Python."""

import numpy as np
import time
from uuid import uuid4

from pyhexz.hexc import CBoard


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
            board.make_move(player, moves[self.rng.integers(0, len(moves))])
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
        b = CBoard(self.board)
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
        return self.board.score()
