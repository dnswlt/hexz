"""Python implementation of the Flagz game.

This file includes the game, MCTS, and neural network implementations.
"""

from bisect import bisect_left
import glob
import h5py
import multiprocessing as mp
import numpy as np
import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from uuid import uuid4

from pyhexz import hexc
from pyhexz.timing import (
    timing,
    timing_ctx,
    clear_perf_stats,
    print_perf_stats,
    print_time,
)
from pyhexz.board import Board, CBoard


class Example:
    """Data holder for one step of a fully played MCTS game."""

    def __init__(
        self,
        game_id: str,
        board: np.ndarray,
        move_probs: np.ndarray,
        turn: int,
        result: int,
        duration_micros: int = None,
    ):
        """
        Args:
            game_id: arbitrary string identifying the game that this example belongs to.
            board: the board as an (N, 11, 10) ndarray.
            move_probs: (2, 11, 10) ndarray of move likelihoods.
            turn: 0 or 1, indicating which player's turn it was.
            result: -1, 0, 1, indicating the player that won (1 = player 0 won).
        """
        self.game_id = game_id
        self.board = board
        self.move_probs = move_probs
        self.turn = turn
        self.result = result
        self.duration_micros = duration_micros

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

    def __init__(self, parent: "Optional[NNode]", player: int, move):
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
        return q + pr[typ, r, c] * np.sqrt(self.parent.visit_count) / (
            1 + self.visit_count
        )

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

    def best_child(self) -> "NNode":
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
        model: HexzNeuralNetwork,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        self.model = model
        self.rng = np.random.default_rng()
        self.device = device
        self.dtype = dtype

        model.eval()  # Training is done outside of NeuralMCTS

    @timing
    @torch.inference_mode()
    def predict(self, board: CBoard, player: int):
        """Predicts move probabilities and value for the given board and player."""
        b = board.b_for(player)
        X = torch.from_numpy(b).to(self.device, dtype=self.dtype)
        pred_pr, pred_val = self.model(torch.unsqueeze(X, 0))
        pred_pr = torch.exp(pred_pr).reshape((2, 11, 10))
        return pred_pr.numpy(force=self.device != "cpu"), pred_val.item()

    @timing
    def run(self, root: hexc.CNN, board: Board) -> None:
        """Runs a single round of the neural MCTS loop."""
        b = Board(board)  # work on a copy.

        # Find leaf node.
        with timing_ctx("run_find_leaf"):
            n = hexc.c_find_leaf(b, root)
        # Reached a leaf node: expand
        player = 1 - n.player()  # Usually it's the other player's turn.
        moves = b.next_moves(player)
        if not moves and n is not root:
            # No more moves for player. Try other player, but not for the first move.
            player = 1 - player
            moves = b.next_moves(player)
        if not moves:
            # Game is over
            n.backpropagate(b.result())
            return
        n.create_children(player, moves)
        # Neural network time! Predict value and policy for next move.
        move_probs, value = self.predict(b, player)
        n.set_move_probs(move_probs)
        n.backpropagate(value)

    def play_game(
        self, board: CBoard, runs_per_move=500, max_moves=200, progress_queue=None
    ) -> list[Example]:
        """Plays one full game and returns the move likelihoods per move and the final result.

        Args:
            runs_per_move: number of MCTS runs to make per move. The first six moves get twice that number of moves,
                because they are the most critical for the whole game.
        """
        game_id = uuid4().hex[:12]
        examples: list[Example] = []
        result = None
        started = time.perf_counter()
        n = 0
        player = 0  # Player 0 always starts.
        root: hexc.CNN = hexc.CNN(None, 1 - player, (0, 0, 0, 0.0))
        for _ in range(max_moves):
            t_start = time.perf_counter_ns()
            # Root's children have the player whose turn it actually is.
            # So root has the opposite player.
            for i in range(runs_per_move if n >= 6 else 2 * runs_per_move):
                self.run(root, board)
            best_child = root.best_child()
            if not best_child:
                # Game over
                result = board.result()
                break
            best_child_player = best_child.player()
            duration_ns = time.perf_counter_ns() - t_start
            # Record example. Examples always contain the board as seen by the player whose turn it is.
            examples.append(
                Example(
                    game_id,
                    board.b_for(best_child_player).copy(),
                    root.move_likelihoods(),
                    best_child_player,
                    None,
                    duration_micros=duration_ns // 1000,
                )
            )
            if progress_queue:
                progress_queue.put(
                    {
                        "examples": 1,
                    }
                )
            # Make the move.
            board.make_move(best_child_player, best_child.move())
            player = 1 - best_child_player
            if n < 5 or n % 10 == 0:
                print(
                    f"Iteration {n} @{time.perf_counter() - started:.3f}s: {best_child=}, score:{board.score()}",
                )
            root = best_child
            root.make_root()
            n += 1
        elapsed = time.perf_counter() - started
        if result is None:
            print(
                f"Game aborted early after {n} moves. Current score: {board.score()}."
            )
            return []  # No examples without a result.

        print(f"Done in {elapsed:.3f}s after {n} moves. Final score: {board.score()}.")
        # Update all examples with result.
        for ex in examples:
            ex.result = result
        return examples


def load_model(path, map_location="cpu"):
    model = HexzNeuralNetwork()
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model


def time_gameplay(device):
    clear_perf_stats()
    b = Board()
    model = HexzNeuralNetwork().to(device)
    m = NeuralMCTS(b, model, device=device)
    _ = m.play_game(runs_per_move=800)
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
    print(
        f"W{worker_id}: Appending game examples to {examples_file}. {args.runs_per_move=}"
    )
    while time.time() - started < args.max_seconds and num_games < args.max_games:
        b = Board()
        m = NeuralMCTS(b, model, device=device)
        examples = m.play_game(
            runs_per_move=args.runs_per_move, progress_queue=progress_queue
        )
        Example.save_all(examples_file, examples)
        num_games += 1
        progress_queue.put(
            {
                "games": 1,
                "done": False,
            }
        )
    progress_queue.put(
        {
            "done": True,
        }
    )
    print_perf_stats()


def record_examples_mp(args):
    num_workers = args.num_workers
    if num_workers <= 0:
        raise ValueError("num_workers must be greater than 0")
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
        if msg.get("done", False):
            running -= 1
        num_examples += msg.get("examples", 0)
        num_games += msg.get("games", 0)
        elapsed = time.perf_counter() - started
        print(
            f"At {elapsed:.1f}s: examples:{num_examples} games:{num_games} ({num_examples/elapsed:.1f} examples/s)."
        )
    for p in procs:
        p.join()


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


def export_model(args):
    model = HexzNeuralNetwork()
    path = args.model
    if os.path.isdir(path):
        path = os.path.join(path, "model.pt")
    if not args.force and os.path.isfile(path):
        print(f"export_model: error: a model already exists at {path}.")
        return
    # Make sure to put the model into eval mode before creating the ScriptModule.
    # At least a model exported with torch.jit.trace cannot be put back into
    # eval mode! (Not sure about the torch.jit.script case).
    model.eval()
    sm = torch.jit.script(model)
    sm.save(path)
    print(
        f"Exported randomly initialized ScriptModule with {sum(p.numel() for p in model.parameters())} parameters to {path}."
    )


def print_model_info(args):
    model_path = path_to_latest_model(args.model)
    model = load_model(model_path)
    print(model)
