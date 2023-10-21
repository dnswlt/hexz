"""sconv converts a zip file of MCTSExamples to an HDF5 file."""

import argparse
import h5py
import numpy as np
import os.path
import sys
import zipfile

from pyhexz.board import Board
from pyhexz.hexz import Example
from pyhexz import hexz_pb2
from pyhexz import svg


def convert_board(proto_board: hexz_pb2.Board, dtype=np.float32) -> np.ndarray:
    board = np.zeros((9, 11, 10), dtype=dtype)
    # Mark invalid cells as blocked
    board[2, [1, 3, 5, 7, 9], 9] = 1
    board[6, [1, 3, 5, 7, 9], 9] = 1
    r, c = 0, 0
    for f in proto_board.flat_fields:
        p = f.owner - 1
        if f.type == hexz_pb2.Field.NORMAL:
            if f.value > 0:
                board[1 + 4 * p, r, c] = f.value
                board[2, r, c] = 1
                board[6, r, c] = 1
            else:
                # Empty cell, might have next_val.
                if f.next_val[0] > 0:
                    board[3, r, c] = f.next_val[0]
                if f.next_val[1] > 0:
                    board[7, r, c] = f.next_val[1]
                # Might also be blocked.
                if f.blocked & 1:
                    board[2, r, c] = 1
                if f.blocked & 2:
                    board[6, r, c] = 1
        elif f.type == hexz_pb2.Field.FLAG:
            board[4 * p, r, c] = 1
            board[2, r, c] = 1
            board[6, r, c] = 1
        elif f.type == hexz_pb2.Field.GRASS:
            board[8, r, c] = f.value
            board[2, r, c] = 1
            board[6, r, c] = 1
        elif f.type == hexz_pb2.Field.ROCK:
            board[2, r, c] = 1
            board[6, r, c] = 1
        else:
            raise ValueError(
                "Unknown cell type: " + hexz_pb2.Field.CellType.Name(f.type)
            )
        c += 1
        if c == 10 - r % 2:
            r += 1
            c = 0
    return board


def convert(e: hexz_pb2.MCTSExample, validate=True, dtype=np.float32) -> Example:
    board = convert_board(e.board, dtype=dtype)
    if validate:
        Board.from_numpy(board).validate()
    move_probs = np.zeros((2, 11, 10), dtype=dtype)
    for s in e.move_stats:
        move_probs[int(s.move.cell_type == 0), s.move.row, s.move.col] = s.visits
    move_probs /= np.sum(move_probs)
    # Players in the neural net are 0 and 1, but in hexz_pb2 they are 1 and 2.
    turn = e.board.turn - 1
    result = np.sign(e.result[0] - e.result[1])
    return Example(e.game_id, board, move_probs, turn, result)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hexz Go example to NeuralMCTS example converter."
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="hdf5",
        help="Format of the generated output",
        choices=("hdf5", "html"),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of examples to convert."
    )
    parser.add_argument("files", type=str, nargs="+", help="Input files to convert.")
    parser.add_argument(
        "--draw-probs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw move probabilities instead of next values (HTML export only).",
    )
    return parser.parse_args()


def convert_to_html(args) -> None:
    for filename in args.files:
        with zipfile.ZipFile(filename, "r") as zf:
            examples = []
            outpath = os.path.splitext(filename)[0] + ".html"
            for i, zi in enumerate(zf.infolist()):
                if i >= args.limit:
                    break
                with zf.open(zi) as f:
                    data = f.read()
                    e = hexz_pb2.MCTSExample()
                    e.ParseFromString(data)
                    examples.append(convert(e))
            svg.export(
                outpath,
                boards=[Board.from_numpy(e.board) for e in examples],
                captions=[e.game_id for e in examples],
                move_probs=[e.move_probs for e in examples]
                if args.draw_probs
                else None,
            )


def convert_to_hdf5(args) -> None:
    dtype = np.float32
    for filename in args.files:
        num_examples = 0
        with zipfile.ZipFile(filename, "r") as zf:
            outpath = os.path.splitext(filename)[0] + ".h5"
            with h5py.File(outpath, "w") as h5:
                for i, zi in enumerate(zf.infolist()):
                    with zf.open(zi) as f:
                        data = f.read()
                        e = hexz_pb2.MCTSExample()
                        e.ParseFromString(data)
                        ex = convert(e, dtype=dtype)
                        grp = h5.create_group(f"{i:08}")
                        grp.attrs["game_id"] = ex.game_id
                        grp.create_dataset("board", data=ex.board)
                        grp.create_dataset("move_probs", data=ex.move_probs)
                        grp.create_dataset(
                            "turn", data=np.array([ex.turn], dtype=dtype)
                        )
                        grp.create_dataset(
                            "result", data=np.array([ex.result], dtype=dtype)
                        )
                        num_examples += 1
            print(f"Converted {num_examples} examples to {outpath}")


def main():
    args = parse_args()
    if any(not f.endswith(".zip") for f in args.files):
        print("Error: all input files must be zip files")
        sys.exit(1)
    if args.output_format == "html":
        convert_to_html(args)
        return
    elif args.output_format == "hdf5":
        convert_to_hdf5(args)
        return
    else:
        print("Unsupported output format: " + args.output_format)
        sys.exit(1)


if __name__ == "__main__":
    main()
