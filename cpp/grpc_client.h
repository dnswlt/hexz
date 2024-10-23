#pragma once

#include <absl/status/status.h>
#include <torch/torch.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "grpcpp/grpcpp.h"
#include "hexz.grpc.pb.h"
#include "hexz.pb.h"

namespace hexz {

// Virtual base class for training service clients.
class TrainingServiceClient {
 public:
  template <typename T>
  using StreamCallback = std::function<void(T)>;
  virtual ~TrainingServiceClient() = default;
  virtual absl::StatusOr<hexzpb::AddTrainingExamplesResponse>
  AddTrainingExamples(const hexzpb::AddTrainingExamplesRequest& request) = 0;

  // Fetches the latest model from the connected training server.
  // Pass an empty model_name to let the training server decide which model to
  // return.
  virtual absl::StatusOr<std::pair<hexzpb::ModelKey, torch::jit::Module>>
  FetchLatestModel(const std::string& model_name) = 0;

  // Makes a streaming RPC call to the training server using the provided
  // context. Each streamed response will be
  virtual absl::Status StreamControlEvents(
      grpc::ClientContext& context, const hexzpb::ControlRequest& request,
      StreamCallback<hexzpb::ControlEvent> callback) {
    return absl::UnimplementedError("StreamControlEvents not implemented");
  }

  virtual std::string RemoteAddr() const = 0;
};

// A gRPC client used to communicate with a training server.
class EmbeddedTrainingServiceClient : public TrainingServiceClient {
 public:
  EmbeddedTrainingServiceClient(std::string path)
      : path_(std::move(path)), model_key_(EmbeddedModelKey()) {}

  absl::StatusOr<hexzpb::AddTrainingExamplesResponse> AddTrainingExamples(
      const hexzpb::AddTrainingExamplesRequest& request) override;

  // Fetches the latest model from the connected training server.
  // Pass an empty model_name to let the training server decide which model to
  // return.
  absl::StatusOr<std::pair<hexzpb::ModelKey, torch::jit::Module>>
  FetchLatestModel(const std::string& model_name) override;

  std::string RemoteAddr() const override { return "<local>"; }

 private:
  hexzpb::ModelKey EmbeddedModelKey() {
    hexzpb::ModelKey key;
    key.set_name("<embedded>");
    key.set_checkpoint(0);
    return key;
  }
  std::string path_;
  const hexzpb::ModelKey model_key_;
};

// A gRPC client used to communicate with a training server.
class GRPCTrainingServiceClient : public TrainingServiceClient {
 public:
  GRPCTrainingServiceClient(std::shared_ptr<grpc::Channel> channel,
                            std::string remote_addr)
      : stub_(hexzpb::TrainingService::NewStub(channel)),
        remote_addr_(std::move(remote_addr)) {}

  // Connects to the training service at the given address addr.
  static std::unique_ptr<GRPCTrainingServiceClient> Connect(
      const std::string& addr);

  absl::StatusOr<hexzpb::AddTrainingExamplesResponse> AddTrainingExamples(
      const hexzpb::AddTrainingExamplesRequest& request) override;

  // Fetches the latest model from the connected training server.
  // Pass an empty model_name to let the training server decide which model to
  // return.
  absl::StatusOr<std::pair<hexzpb::ModelKey, torch::jit::Module>>
  FetchLatestModel(const std::string& model_name) override;

  // Starts a server-side streaming RPC and calls the provided callback
  // with each received ControlEvent.
  // This method blocks until all events have been received or the
  // RPC has been terminated. Callers should call TryCancel on the
  // context to cancel the RPC prematurely.
  absl::Status StreamControlEvents(
      grpc::ClientContext& context, const hexzpb::ControlRequest& request,
      StreamCallback<hexzpb::ControlEvent> callback) override;

  std::string RemoteAddr() const override { return remote_addr_; }

 private:
  std::string remote_addr_;
  std::unique_ptr<hexzpb::TrainingService::Stub> stub_;
};

}  // namespace hexz
