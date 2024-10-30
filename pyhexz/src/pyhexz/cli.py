import gzip
import sys

from pyhexz import hexz_pb2
from google.protobuf import json_format


def print_example(path):
    with gzip.open(path, "rb") as f:
        req = hexz_pb2.AddTrainingExamplesRequest()
        req.ParseFromString(f.read())
    print(f"Execution ID: {req.execution_id}")
    print(f"Examples: {len(req.examples)}")
    if req.examples:
        ex = req.examples[10]
        print("Stats:", json_format.MessageToJson(ex.stats))
        print("Move:", json_format.MessageToJson(ex.move))
        print(f"Predicted value: {ex.model_predictions.value}")


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
