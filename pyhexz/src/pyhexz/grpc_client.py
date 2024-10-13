import io
import grpc
import torch

from pyhexz import hexz_pb2
from pyhexz import hexz_pb2_grpc


def _torch_bytes(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = hexz_pb2_grpc.TrainingServiceStub(channel)
    req = hexz_pb2.AddTrainingExamplesRequest(
        execution_id="test_client",
        examples=[
            hexz_pb2.TrainingExample(
                encoding=hexz_pb2.TrainingExample.Encoding.PYTORCH,
                model_key=hexz_pb2.ModelKey(
                    name="test_model",
                    checkpoint=0,
                ),
                board=_torch_bytes(torch.randn((11, 11, 10))),
                action_mask=_torch_bytes(
                    torch.randn((2, 11, 10)) < 0.5
                ),  # boolean tensor
                move_probs=_torch_bytes(torch.randn((2, 11, 10))),
                result=0.5,
            )
        ],
    )
    resp: hexz_pb2.AddTrainingExamplesResponse = stub.AddTrainingExamples(req)
    status = hexz_pb2.AddTrainingExamplesResponse.Status.Name(resp.status)
    print(f"Received response with status: {status}")


if __name__ == "__main__":
    main()
