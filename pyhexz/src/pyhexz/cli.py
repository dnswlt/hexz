import gzip
from io import BytesIO
import sys
import torch

from pyhexz import hexz_pb2
from google.protobuf import json_format


def print_example(path):
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
        priors = torch.load(BytesIO(ex.model_predictions.priors))
        print(f"Chosen move prior: {priors[idx, ex.move.row, ex.move.col].item()}")
        nonzero_priors = priors[priors.nonzero(as_tuple=True)]
        quartiles = torch.quantile(nonzero_priors, q=torch.tensor([0, 0.25, 0.5, 0.75, 1]))
        print(f"Nonzero prior quartiles (min/25/50/75/max): {quartiles}")


def main(argv) -> int:
    if len(argv) < 2:
        print("Need to specify a command")
        return 1
    if argv[1] == "print_request":
        if len(argv) != 3:
            print(f"Usage: {argv[0]} print_request <request.gz>")
            return 1
        print_example(argv[2])
    else:
        print(f"Unknown command: {argv[1]}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
