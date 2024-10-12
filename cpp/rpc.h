#ifndef __HEXZ_RPC_H__
#define __HEXZ_RPC_H__

#include <absl/status/statusor.h>
#include <torch/torch.h>

#include "base.h"
#include "hexz.pb.h"

namespace hexz {

struct KeyedModel {
  hexzpb::ModelKey key;
  torch::jit::Module model;
};

class RPCClient {
 public:
  RPCClient(std::string training_server_url)
      : training_server_url_{std::move(training_server_url)} {}
  // Fetches the latest model from the server. The model will be on the "cpu"
  // device.
  absl::StatusOr<KeyedModel> FetchLatestModel();

  // Sends the given examples to the training server.
  // Note that the example vector should get moved into this method.
  absl::StatusOr<hexzpb::AddTrainingExamplesResponse> SendExamples(
      const std::string& execution_id,
      std::vector<hexzpb::TrainingExample>&& examples);

 private:
  const std::string training_server_url_;
};

}  // namespace hexz

#endif  // ____HEXZ_RPC_H__
