#pragma once

#include <torch/torch.h>

#include <optional>

#include "hexz.grpc.pb.h"
#include "model.h"

namespace hexz {

struct CPUPlayerServiceConfig {
  std::string model_path;
  hexzpb::ModelKey model_key;
  torch::Device device = torch::kCPU;
  int64_t max_think_time_ms = 0;
};

class CPUPlayerServiceImpl final : public hexzpb::CPUPlayerService::Service {
 public:
  CPUPlayerServiceImpl(CPUPlayerServiceConfig config);

  grpc::Status SuggestMove(grpc::ServerContext* context,
                           const hexzpb::SuggestMoveRequest* request,
                           hexzpb::SuggestMoveResponse* response) override;

 private:
  CPUPlayerServiceConfig config_;
  std::mutex module_mut_;
  TorchModel model_;
};

}  // namespace hexz
