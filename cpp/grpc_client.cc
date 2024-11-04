#include "grpc_client.h"

#include <absl/log/absl_log.h>
#include <absl/status/status.h>
#include <absl/strings/str_cat.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <sstream>

#include "grpcpp/grpcpp.h"

namespace hexz {

namespace {
#include <grpcpp/grpcpp.h>

#include "absl/status/status.h"

// Helper function to translate grpc::StatusCode to absl::StatusCode
absl::StatusCode AbslStatusCode(grpc::StatusCode grpc_code) {
  switch (grpc_code) {
    case grpc::StatusCode::OK:
      return absl::StatusCode::kOk;
    case grpc::StatusCode::CANCELLED:
      return absl::StatusCode::kCancelled;
    case grpc::StatusCode::UNKNOWN:
      return absl::StatusCode::kUnknown;
    case grpc::StatusCode::INVALID_ARGUMENT:
      return absl::StatusCode::kInvalidArgument;
    case grpc::StatusCode::DEADLINE_EXCEEDED:
      return absl::StatusCode::kDeadlineExceeded;
    case grpc::StatusCode::NOT_FOUND:
      return absl::StatusCode::kNotFound;
    case grpc::StatusCode::ALREADY_EXISTS:
      return absl::StatusCode::kAlreadyExists;
    case grpc::StatusCode::PERMISSION_DENIED:
      return absl::StatusCode::kPermissionDenied;
    case grpc::StatusCode::UNAUTHENTICATED:
      return absl::StatusCode::kUnauthenticated;
    case grpc::StatusCode::RESOURCE_EXHAUSTED:
      return absl::StatusCode::kResourceExhausted;
    case grpc::StatusCode::FAILED_PRECONDITION:
      return absl::StatusCode::kFailedPrecondition;
    case grpc::StatusCode::ABORTED:
      return absl::StatusCode::kAborted;
    case grpc::StatusCode::OUT_OF_RANGE:
      return absl::StatusCode::kOutOfRange;
    case grpc::StatusCode::UNIMPLEMENTED:
      return absl::StatusCode::kUnimplemented;
    case grpc::StatusCode::INTERNAL:
      return absl::StatusCode::kInternal;
    case grpc::StatusCode::UNAVAILABLE:
      return absl::StatusCode::kUnavailable;
    case grpc::StatusCode::DATA_LOSS:
      return absl::StatusCode::kDataLoss;
    default:
      return absl::StatusCode::kUnknown;
  }
}

// Function to translate grpc::Status to absl::Status
absl::Status AbslStatus(const grpc::Status& grpc_status) {
  absl::StatusCode code = AbslStatusCode(grpc_status.error_code());
  return absl::Status(code, grpc_status.error_message());
}

}  // namespace

absl::StatusOr<hexzpb::AddTrainingExamplesResponse>
EmbeddedTrainingServiceClient::AddTrainingExamples(
    const hexzpb::AddTrainingExamplesRequest& request) {
  hexzpb::AddTrainingExamplesResponse response;
  response.set_status(hexzpb::AddTrainingExamplesResponse::ACCEPTED);
  *response.mutable_latest_model() = model_key_;
  return response;
}

absl::StatusOr<std::pair<hexzpb::ModelKey, torch::jit::Module>>
EmbeddedTrainingServiceClient::FetchLatestModel(const std::string& model_name) {
  try {
    auto model = torch::jit::load(path_, torch::kCPU);
    return std::make_pair(model_key_, model);

  } catch (c10::Error& error) {
    return absl::InternalError(
        absl::StrCat("torch::jit::load failed: ", error.msg()));
  }
}

absl::StatusOr<hexzpb::AddTrainingExamplesResponse>
GRPCTrainingServiceClient::AddTrainingExamples(
    const hexzpb::AddTrainingExamplesRequest& request) {
  // Data we are sending to the server.
  hexzpb::AddTrainingExamplesResponse resp;

  // Compress the (possibly large, >1M) request.
  grpc::ClientContext context;
  context.set_compression_algorithm(GRPC_COMPRESS_GZIP);

  grpc::Status status = stub_->AddTrainingExamples(&context, request, &resp);

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "AddTrainingExamples RPC failed: Code="
                    << status.error_code()
                    << " Message=" << status.error_message();
    return absl::InternalError("Failed to send examples");
  }
  return resp;
}

absl::StatusOr<std::pair<hexzpb::ModelKey, torch::jit::Module>>
GRPCTrainingServiceClient::FetchLatestModel(const std::string& model_name) {
  hexzpb::FetchModelRequest request;
  if (model_name != "") {
    request.mutable_model_key()->set_name(model_name);
    request.mutable_model_key()->set_checkpoint(-1);  // Always get the lastest.
  }
  request.set_encoding(hexzpb::ModelEncoding::JIT_SCRIPT);

  hexzpb::FetchModelResponse resp;

  grpc::ClientContext context;

  grpc::Status status = stub_->FetchModel(&context, request, &resp);

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "FetchModel RPC failed: Code=" << status.error_code()
                    << " Message=" << status.error_message();
    return absl::InternalError("Failed to fetch model");
  }

  try {
    std::istringstream model_is(resp.model_bytes());
    auto model = torch::jit::load(model_is, torch::kCPU);
    return std::make_pair(resp.model_key(), model);
  } catch (const c10::Error& e) {
    return absl::InternalError(
        absl::StrCat("Failed to torch::jit::load module: ", e.msg()));
  }
}

absl::Status GRPCTrainingServiceClient::StreamControlEvents(
    grpc::ClientContext& context,
    const hexzpb::ControlRequest& request,
    StreamCallback<hexzpb::ControlEvent> callback) {
  std::unique_ptr<grpc::ClientReader<hexzpb::ControlEvent>> reader(
      stub_->ControlEvents(&context, request));
  hexzpb::ControlEvent event;
  while (reader->Read(&event)) {
    callback(event);
  }
  return AbslStatus(reader->Finish());
}

std::unique_ptr<GRPCTrainingServiceClient> GRPCTrainingServiceClient::Connect(
    const std::string& addr) {
  grpc::ChannelArguments channel_args;
  channel_args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
  channel_args.SetMaxReceiveMessageSize(256<<20); // 256MiB

  return std::make_unique<GRPCTrainingServiceClient>(
      grpc::CreateCustomChannel(addr, grpc::InsecureChannelCredentials(),
                                channel_args),
      addr);
}

}  // namespace hexz
