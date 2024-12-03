#pragma once

#include <torch/torch.h>

#include <optional>

#include "hexz.grpc.pb.h"
#include "model.h"

namespace hexz {

// Maximum number of items in a SuggestMovesRequest.
// Requests containing more items will be rejected.
inline constexpr int kMaxRequestBatchSize = 128;

struct CPUPlayerServiceConfig {
  std::string model_path;
  hexzpb::ModelKey model_key;
  torch::DeviceType device_type = torch::kCPU;
  int64_t max_think_time_ms = 0;
  int batch_size = 16;
};

// Implementation of the gRPC server for the CPUPlayerService.
class CPUPlayerServiceImpl final : public hexzpb::CPUPlayerService::Service {
 public:
  CPUPlayerServiceImpl(CPUPlayerServiceConfig config);

  grpc::Status ServerInfo(grpc::ServerContext* context,
                          const hexzpb::ServerInfoRequest* request,
                          hexzpb::ServerInfoResponse* response) override;

  grpc::Status SuggestMove(grpc::ServerContext* context,
                           const hexzpb::SuggestMoveRequest* request,
                           hexzpb::SuggestMoveResponse* response) override;

  grpc::Status SuggestMoves(grpc::ServerContext* context,
                            const hexzpb::SuggestMovesRequest* request,
                            hexzpb::SuggestMovesResponse* response) override;

 private:
  absl::StatusOr<hexzpb::SuggestMoveResponse> DoSuggestMove(
      const hexzpb::GameEngineState& state, int64_t max_think_time_ms,
      int64_t max_iterations);

  CPUPlayerServiceConfig config_;
  std::mutex module_mut_;
  FiberTorchModel model_;
};

}  // namespace hexz
