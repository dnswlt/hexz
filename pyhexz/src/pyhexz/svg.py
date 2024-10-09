"""SVG export of boards. Pretty much a direct translation of the svg.go code to Python."""

import io
import math
from typing import Any, Optional
import typing
import numpy as np
import time

import torch

from pyhexz import hexz_pb2
from pyhexz.board import Board


def svg_hexagon(
    side_length: float, board: Board, r: int, c: int, move_probs: np.ndarray = None
) -> str:
    flag_path = "M 3.3072 -38.5457 C -5.0739 -45.2506 -14.9125 -45.0416 -21.9362 -43.5629 C -26.2188 -42.6613 -30.5708 -41.1876 -34.3913 -39.0083 C -36.0409 -38.0657 -37.0588 -36.3116 -37.0588 -34.4117 V 39.7059 C -37.0588 42.6298 -34.6885 45 -31.7647 45 C -28.8409 45 -26.4706 42.6298 -26.4706 39.7059 V 11.2733 C -19.185 8.2022 -9.9318 6.7759 -3.3072 12.0754 C 5.0739 18.7798 14.9125 18.5712 21.9362 17.0926 C 27.2959 15.9639 31.4688 14.0405 33.3286 13.1014 C 35.4012 12.0552 37.0588 10.4474 37.0588 7.9412 V -34.4117 C 37.0588 -36.299 36.054 -38.0434 34.4218 -38.9907 C 32.7907 -39.9373 30.7821 -39.9461 29.1441 -39.0116 L 29.1393 -39.0091 C 21.5264 -34.8494 10.5951 -32.7153 3.3072 -38.5457 Z M -26.4706 -31.0797 V -0.0053 C -16.5104 -3.1934 -5.1829 -2.9848 3.3072 3.807 C 8.1614 7.6908 14.2052 7.8999 19.755 6.7315 C 22.4132 6.1714 24.7601 5.3301 26.4706 4.6091 V -26.465 C 25.0888 -26.0227 23.5646 -25.6034 21.9362 -25.2606 C 14.9125 -23.7819 5.0739 -23.5729 -3.3072 -30.2777 C -9.9318 -35.5772 -19.185 -34.1507 -26.4706 -31.0797 Z"
    icon_colors = {
        0: "#1e1e1e",
        1: "#7a6505",
    }
    a = side_length
    b = math.sqrt(3) * a
    icon_scale = b / 50 / 3.5  # Just about right.
    ps = [
        b / 2,
        a / 2,
        0,
        a,
        -b / 2,
        a / 2,
        -b / 2,
        -a / 2,
        0,
        -a,
        b / 2,
        -a / 2,
    ]
    coords = [f"{x:.6f},{y:.6f}" for x, y in zip(ps[::2], ps[1::2])]
    draw_next_value = move_probs is None
    if board.b[0, r, c] > 0 or board.b[1, r, c] > 0:
        fill = "#255ab4"
    elif board.b[5, r, c] > 0 or board.b[6, r, c] > 0:
        fill = "#f8d748"
    elif board.b[10, r, c] > 0:
        # Grass
        fill = "#008048"
    elif board.b[2, r, c] > 0 and board.b[7, r, c] > 0:
        # Blocked for both: rock
        fill = "#5f5f5f"
    elif board.b[2, r, c] > 0:
        # Blocked only for player 0
        fill = "#fbeba3"
    elif board.b[7, r, c] > 0:
        # Blocked only for player 1
        fill = "#92acd9"
    elif draw_next_value and board.b[3, r, c] > 0 and board.b[8, r, c] > 0:
        # Next value for both players.
        fill = "#B5CB99"
    elif draw_next_value and board.b[3, r, c] > 0:
        # Next value only for player 0
        fill = "#92acd9"
    elif draw_next_value and board.b[8, r, c] > 0:
        # Next value only for player 1
        fill = "#fbeba3"
    else:
        fill = "none"
    points = " ".join(coords)
    x_off = 0.0
    if r % 2 == 1:
        x_off = b / 2
    x = x_off + c * b + b / 2
    y = r * a * 3 / 2 + a
    transform = f"translate({x:.6f},{y:.6f})"
    elems = [
        f'<polygon points="{points}" stroke="#cbcbcb" stroke-width="1" fill="{fill}" />'
    ]
    if board.b[0, r, c] > 0 or board.b[5, r, c] > 0:
        # Flag of one of the players.
        player = int(board.b[5, r, c])
        elems.append(
            f'<g transform="scale({icon_scale:.6f})"><path d="{flag_path}" fill-rule="evenodd" fill="{icon_colors[player]}" /></g>'
        )
    elif board.b[1, r, c] > 0 or board.b[6, r, c] > 0:
        # Occupied by one of the players.
        player = int(board.b[6, r, c] > 0)
        value = int(max(board.b[1, r, c], board.b[6, r, c]))
        elems.append(
            f'<text style="text-anchor: middle; alignment-baseline: middle; font: {int(a)}px sans-serif;" fill="{icon_colors[player]}" x="0" y="0">{value}</text>'
        )
    elif board.b[10, r, c] > 0:
        # Grass
        value = int(board.b[10, r, c])
        elems.append(
            f'<text style="text-anchor: middle; alignment-baseline: middle; font: {int(a)}px sans-serif;" fill="#1e1e1e" x="0" y="0">{value}</text>'
        )
    elif draw_next_value and (board.b[3, r, c] > 0 or board.b[8, r, c] > 0):
        # Next value. Only drawn if we don't also draw move_probs.
        if board.b[3, r, c] > 0 and board.b[8, r, c] > 0:
            # both players have a next value
            value = f"{int(board.b[3, r, c])}&middot;{int(board.b[8, r, c])}"
            elems.append(
                f'<text style="text-anchor: middle; alignment-baseline: middle; font: {int(a*.8)}px sans-serif;" fill="#008048" x="0" y="0">{value}</text>'
            )
        else:
            player = int(board.b[8, r, c] > 0)
            value = int(max(board.b[3, r, c], board.b[8, r, c]))
            elems.append(
                f'<text style="text-anchor: middle; alignment-baseline: middle; font: {int(a)}px sans-serif;" fill="{icon_colors[player]}" x="0" y="0">{value}</text>'
            )
    elif move_probs is not None and move_probs[:, r, c].max() > 0:
        # Move probabilities
        elems.append(
            f"""<text style="text-anchor: middle; alignment-baseline: middle; font: {int(a/2)}px sans-serif;" fill="#cbcbcb" x="0" y="0">
                           <tspan x="0" dy="0">{100*move_probs[0, r, c]:.1f}</tspan>
                           <tspan x="0" dy="1.2em">{100*move_probs[1, r, c]:.1f}</tspan>
                         </text>"""
        )

    return f'<g transform="{transform}">{"".join(elems)}</g>'


class NumpyExample(typing.NamedTuple):
    board: np.ndarray  # shape: (11, 11, 10)
    action_mask: np.ndarray  # shape: (2, 11, 10)
    move_probs: np.ndarray  # shape: (2, 11, 10)
    value: np.ndarray  # shape: (1,)
    priors: np.ndarray  # shape: (2, 11, 10)

    @classmethod
    def decode(cls, ex: hexz_pb2.TrainingExample) -> "NumpyExample":
        """Decodes the given TrainingExample and returns its data as a named tuple of np arrays."""
        priors = None  # Optional.
        if ex.encoding == hexz_pb2.TrainingExample.PYTORCH:
            board = torch.load(io.BytesIO(ex.board), weights_only=True).numpy()
            action_mask = torch.load(io.BytesIO(ex.action_mask), weights_only=True).numpy()
            pr = torch.load(io.BytesIO(ex.move_probs), weights_only=True).numpy()
            if ex.model_predictions.priors:
                priors = torch.load(io.BytesIO(ex.model_predictions.priors), weights_only=True).numpy()
        else:
            board = np.load(io.BytesIO(ex.board))
            action_mask = np.load(io.BytesIO(ex.action_mask))
            pr = np.load(io.BytesIO(ex.move_probs))
            if ex.model_predictions.priors:
                priors = np.load(io.BytesIO(ex.model_predictions.priors))
        if board.shape != (11, 11, 10):
            raise ValueError(f"Wrong board shape: {board.shape}.")
        if action_mask.shape != (2, 11, 10):
            raise ValueError(f"Wrong action_mask shape: {pr.shape}.")
        if pr.shape != (2, 11, 10):
            raise ValueError(f"Wrong move_probs shape: {pr.shape}.")
        if priors is not None and priors.shape != (2, 11, 10):
            raise ValueError(f"Wrong model_predictions.priors shape: {priors.shape}.")
        val = np.array([ex.result], dtype=np.float32)
        return NumpyExample(
            board=board,
            action_mask=action_mask,
            move_probs=pr,
            value=val,
            priors=priors,
        )


def export(
    file_like: Any,
    req: hexz_pb2.AddTrainingExamplesRequest,
) -> None:
    side_length = 30.0
    width = 10 * math.sqrt(3) * side_length
    height = 17 * side_length
    viewbox = f"0 0 {width:.6f} {height:.6f}"
    f = file_like
    f.write(
        """<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Hexz SVG Export</title>
        <style>
            body {
                background-color: #1e1e1e;
                color: #cbcbcb;
            }
            body {
                font-family: sans-serif;
            }
        </style>
    </head>
    <body>
        <h1>Hexz SVG Export</h1>\n"""
    )
    f.write(f"<p>Created: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
    f.write(f"<p>Execution ID: {req.execution_id}</p>")
    for i, ex in enumerate(req.examples):
        f.write(f"<h2>Board {i}</h2>\n")
        n = NumpyExample.decode(ex)
        board = Board.from_numpy(n.board)
        p0, p1 = board.score()
        f.write(f"<p>Score: {int(p0)} &ndash; {int(p1)}</p>\n")
        f.write("<div>\n")
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.6f}" height="{height:.6f}" viewBox="{viewbox}">\n'
        )
        for r in range(11):
            for c in range(10 - r % 2):
                hex = svg_hexagon(side_length, board, r, c, n.move_probs)
                f.write(hex + "\n")
        f.write("</svg>\n")
        caption = f"Move: {ex.move.move} &bull; Value: {n.value[0]:.3f} PredVal: {ex.model_predictions.value:.3f}"
        f.write(f"<p>{caption}</p>\n")
        f.write("</div>\n")
    f.write("</body>\n")
    f.write("</html>\n")
