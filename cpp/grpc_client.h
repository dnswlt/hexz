#include <torch/torch.h>

#include <memory>
#include <string>
#include <utility>

#include "grpcpp/grpcpp.h"
#include "hexz.grpc.pb.h"
#include "hexz.pb.h"

namespace hexz {

// A gRPC client used to communicate with a training server.
class TrainingServiceClient {
 public:
  TrainingServiceClient(std::shared_ptr<grpc::Channel> channel)
      : stub_(hexzpb::TrainingService::NewStub(channel)) {}

  static std::unique_ptr<TrainingServiceClient> MustConnect(
      const std::string& addr);
      
  absl::StatusOr<hexzpb::AddTrainingExamplesResponse> AddTrainingExamples(
      const hexzpb::AddTrainingExamplesRequest& request);

  absl::StatusOr<std::pair<hexzpb::ModelKey, torch::jit::Module>>
  FetchLatestModel(const std::string& model_name);

 private:
  std::unique_ptr<hexzpb::TrainingService::Stub> stub_;
};

}  // namespace hexz
