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

absl::StatusOr<hexzpb::AddTrainingExamplesResponse>
EmbeddedTrainingServiceClient::AddTrainingExamples(
    const hexzpb::AddTrainingExamplesRequest& request) {
  hexzpb::AddTrainingExamplesResponse response;
  response.set_status(hexzpb::AddTrainingExamplesResponse::ACCEPTED);
  return response;
}

absl::StatusOr<std::pair<hexzpb::ModelKey, torch::jit::Module>>
EmbeddedTrainingServiceClient::FetchLatestModel(const std::string& model_name) {
  hexzpb::ModelKey key;
  key.set_name("embedded");
  try {
    auto model = torch::jit::load(path_, torch::kCPU);
    return std::make_pair(key, model);

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

std::unique_ptr<GRPCTrainingServiceClient> GRPCTrainingServiceClient::Connect(
    const std::string& addr) {
  grpc::ChannelArguments channel_args;
  channel_args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);

  return std::make_unique<GRPCTrainingServiceClient>(grpc::CreateCustomChannel(
      addr, grpc::InsecureChannelCredentials(), channel_args));
}

}  // namespace hexz
