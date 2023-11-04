#include <absl/base/log_severity.h>
#include <absl/log/absl_log.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <algorithm>
#include <cassert>
#include <ctime>
#include <iostream>
#include <sstream>
#include <utility>

#include "base.h"
#include "board.h"
#include "hexz.pb.h"
#include "mcts.h"
#include "perfm.h"
#include "rpc.h"

namespace hexz {

std::string ModelId(const hexzpb::ModelKey& key) {
  return absl::StrCat(key.name(), ":", key.checkpoint());
}

void PlayGameLocally(const Config& config) {
  torch::jit::script::Module model;
  try {
    model = torch::jit::load(config.local_model_path);
    model.to(torch::kCPU);
    model.eval();
  } catch (const c10::Error& e) {
    ABSL_LOG(ERROR) << "Failed to load model from " << config.local_model_path
                    << ": " << e.msg();
    return;
  }
  {
    Perfm::Scope ps(Perfm::PlayGameLocally);
    NeuralMCTS mcts{model, config};
    Board b = Board::RandomBoard();
    if (auto result = mcts.PlayGame(b, /*max_runtime_seconds=*/0);
        !result.ok()) {
      ABSL_LOG(ERROR) << "Game failed: " << result.status();
    }
  }
}

void GenerateExamples(const Config& config) {
  const auto started_micros = UnixMicros();
  RPCClient rpc(config);
  auto km = rpc.FetchLatestModel();
  if (!km.ok()) {
    ABSL_LOG(ERROR) << "FetchModule failed: " << km.status();
    return;
  }
  int max_games =
      config.max_games > 0 ? config.max_games : std::numeric_limits<int>::max();
  for (int i = 0; i < max_games; i++) {
    auto now = UnixMicros();
    if (now >= started_micros + config.max_runtime_seconds * 1'000'000) {
      break;
    }
    NeuralMCTS mcts(km->model, config);
    Board b = Board::RandomBoard();
    int max_runtime_seconds =
        config.max_runtime_seconds - (now - started_micros) / 1'000'000;

    auto examples = mcts.PlayGame(b, max_runtime_seconds);

    if (!examples.ok()) {
      if (absl::IsDeadlineExceeded(examples.status())) {
        break;
      }
      ABSL_LOG(ERROR) << "Aborting: PlayGame returned an error: "
                      << examples.status();
      return;
    }
    const int n_examples = examples->size();
    auto resp = rpc.SendExamples(km->key, *std::move(examples));
    if (!resp.ok()) {
      ABSL_LOG(ERROR) << "Failed to send examples: " << resp.status();
      return;
    }
    if (resp->status() ==
        hexzpb::AddTrainingExamplesResponse::REJECTED_AT_CAPACITY) {
      // For now, immediately exit if server is at capacity.
      // There are too many workers sending examples.
      ABSL_LOG(ERROR) << "Server is at capacity: " << resp->error_message();
      return;
    }
    if (resp->status() ==
        hexzpb::AddTrainingExamplesResponse::REJECTED_WRONG_MODEL) {
      // Since the server should accept both the previous and the latest model
      // version, this should be a rare event, and we should probably tune our
      // batch size. Individual workers are not able to generate new examples
      // before the model got updated twice.
      ABSL_LOG(ERROR)
          << "Server rejected out examples due to an old model. Sent: "
          << ModelId(km->key) << ", want: " << ModelId(resp->latest_model());
    }
    if (resp->status() != hexzpb::AddTrainingExamplesResponse::ACCEPTED) {
      ABSL_LOG(ERROR) << "Server did not like our examples: "
                      << hexzpb::AddTrainingExamplesResponse::Status_Name(
                             resp->status())
                      << ": " << resp->error_message();
      return;
    }
    // Check if we need to update to a new model.
    if (resp->latest_model().name() != km->key.name() ||
        resp->latest_model().checkpoint() != km->key.checkpoint()) {
      km = rpc.FetchLatestModel();
      if (!km.ok()) {
        ABSL_LOG(ERROR) << "FetchModule failed: " << km.status();
        return;
      }
    }
    ABSL_LOG(INFO) << "Successfully sent " << n_examples
                   << " examples to training server at "
                   << config.training_server_url;
  }
}

}  // namespace hexz

int main() {
  // Initialization
  hexz::Perfm::InitScope perfm;
  // absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  // Config
  auto config = hexz::Config::FromEnv();
  hexz::Node::uct_c = config.uct_c;

  // Execute
  ABSL_LOG(INFO) << "Worker started with " << config.String();
  if (config.local_model_path != "") {
    ABSL_LOG(INFO)
        << "HEXZ_LOCAL_MODEL_PATH is set. Playing a game with a local model.";
    PlayGameLocally(config);
    return 0;
  }
  if (config.training_server_url != "") {
    ABSL_LOG(INFO)
        << "Generating examples and sending them to the training server.";
    hexz::GenerateExamples(config);
    return 0;
  }
  return 0;
}
