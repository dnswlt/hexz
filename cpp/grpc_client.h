#ifndef __HEXZ_GRPC_CLIENT_H__
#define __HEXZ_GRPC_CLIENT_H__

#include <torch/torch.h>

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
  virtual ~TrainingServiceClient() = default;
  virtual absl::StatusOr<hexzpb::AddTrainingExamplesResponse>
  AddTrainingExamples(const hexzpb::AddTrainingExamplesRequest& request) = 0;

  // Fetches the latest model from the connected training server.
  // Pass an empty model_name to let the training server decide which model to
  // return.
  virtual absl::StatusOr<std::pair<hexzpb::ModelKey, torch::jit::Module>>
  FetchLatestModel(const std::string& model_name) = 0;
};

// A gRPC client used to communicate with a training server.
class EmbeddedTrainingServiceClient : public TrainingServiceClient {
 public:
  EmbeddedTrainingServiceClient(std::string path) : path_(std::move(path)) {}

  absl::StatusOr<hexzpb::AddTrainingExamplesResponse> AddTrainingExamples(
      const hexzpb::AddTrainingExamplesRequest& request) override;

  // Fetches the latest model from the connected training server.
  // Pass an empty model_name to let the training server decide which model to
  // return.
  absl::StatusOr<std::pair<hexzpb::ModelKey, torch::jit::Module>>
  FetchLatestModel(const std::string& model_name) override;

 private:
  std::string path_;
};

// A gRPC client used to communicate with a training server.
class GRPCTrainingServiceClient : public TrainingServiceClient {
 public:
  GRPCTrainingServiceClient(std::shared_ptr<grpc::Channel> channel)
      : stub_(hexzpb::TrainingService::NewStub(channel)) {}

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

 private:
  std::unique_ptr<hexzpb::TrainingService::Stub> stub_;
};

}  // namespace hexz

#endif  // __HEXZ_GRPC_CLIENT_H__