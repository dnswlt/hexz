#pragma once

#include <torch/torch.h>

#include <optional>
#include <semaphore>

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
  int max_batch_size = 128;
  // Maximum number of concurrent requests that the server will accept.
  // Any further request will be denied with a RESOURCE_EXHAUSTED error.
  // SuggestMovesRequest count as N multiple requests, where N is the
  // number of contained game engine states.
  int max_concurrent_requests = 128;
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

 private:
  absl::StatusOr<hexzpb::SuggestMoveResponse> DoSuggestMove(
      const hexzpb::GameEngineState& state, int64_t max_think_time_ms,
      int64_t max_iterations);

  CPUPlayerServiceConfig config_;
  std::mutex module_mut_;
  FiberTorchModel model_;
  std::counting_semaphore<> concurrent_rpc_sem_;
};

}  // namespace hexz
