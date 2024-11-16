import argparse
import gzip
from io import BytesIO
import os
import sys
import torch

from pyhexz import hexz_pb2, svg
from google.protobuf import json_format

from pyhexz.board import Board
from pyhexz.model import HexzNeuralNetwork


def print_request(args):
    path = args.request_file
    with gzip.open(path, "rb") as f:
        req = hexz_pb2.AddTrainingExamplesRequest()
        req.ParseFromString(f.read())
    print(f"Worker Config: {json_format.MessageToJson(req.worker_config)}")
    print(f"Examples: {len(req.examples)}")
    if req.examples:
        ex = req.examples[0]
        print("Stats:", json_format.MessageToJson(ex.stats))
        print("Move:", json_format.MessageToJson(ex.move))
        print(f"Model key: {ex.model_key.name}:{ex.model_key.checkpoint}")
        print(f"Predicted value: {ex.model_predictions.value}")
        idx = 0 if ex.move.cell_type == hexz_pb2.Field.CellType.FLAG else 1
        priors = torch.load(BytesIO(ex.model_predictions.priors), weights_only=True)
        print(f"Chosen move prior: {priors[idx, ex.move.row, ex.move.col].item()}")
        nonzero_priors = priors[priors.nonzero(as_tuple=True)]
        quartiles = torch.quantile(
            nonzero_priors, q=torch.tensor([0, 0.25, 0.5, 0.75, 1])
        )
        print(f"Nonzero prior quartiles (min/25/50/75/max): {quartiles}")
    ex_n = req.examples[-1]
    b = Board.from_numpy(torch.load(BytesIO(ex_n.board), weights_only=True).numpy())
    s1, s2 = b.score()
    print(f"Score of last example: {int(s1)}-{int(s2)}")


def html_export(args):
    with gzip.open(args.request_file, "rb") as f:
        req = hexz_pb2.AddTrainingExamplesRequest()
        req.ParseFromString(f.read())
    with open(args.out_path, 'w') as f_out:
        svg.export(f_out, req)


def create_model(args) -> int:
    network_args = {
        key: getattr(args, key) for key in ["blocks", "filters", "model_type"]
    }
    model = HexzNeuralNetwork(**network_args)
    scriptmodule = torch.jit.script(model)
    scriptmodule.save(args.outfile)
    size = os.path.getsize(args.outfile)
    print(
        f"Saved PyTorch scriptmodule {network_args} ({size:,} bytes) to {args.outfile}"
    )
    return 0


def main(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="pyhexzx.cli", description="Utility CLI for pyhexz"
    )

    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    # Subcommand: create_model
    create_parser = subparsers.add_parser("create_model", help="Create a model")
    create_parser.add_argument(
        "--model_type",
        required=True,
        choices=["conv2d", "resnet"],
        help="Type of the model",
    )
    create_parser.add_argument(
        "--filters", type=int, default=128, help="Number of CNN filters for the model"
    )
    create_parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="Number of blocks (residual or CNN) for the model",
    )
    create_parser.add_argument("outfile", help="Path to write the scriptmodule to")
    create_parser.set_defaults(func=create_model)

    # Subcommand: print_model
    print_parser = subparsers.add_parser(
        "print_request", help="Print AddTrainingExamplesRequest details"
    )
    print_parser.add_argument("request_file", help="Path of the request to print")
    print_parser.set_defaults(func=print_request)

    # Subcommand: html_export
    export_parser = subparsers.add_parser(
        "html_export", help="Export examples from AddTrainingExamplesRequest to HTML"
    )
    export_parser.add_argument(
        "--out_path",
        type=str,
        default="./export.html",
        help="Path of the generated HTML export",
    )
    export_parser.add_argument("request_file", help="Path of the request to export")
    export_parser.set_defaults(func=html_export)

    # Parse and call the appropriate function based on the command
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
