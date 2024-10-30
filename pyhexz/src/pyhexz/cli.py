import gzip
import sys

from pyhexz import hexz_pb2


def print_example(path):
    with gzip.open(path, "rb") as f:
        req = hexz_pb2.AddTrainingExamplesRequest()
        req.ParseFromString(f.read())
    print(req.execution_id)

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
