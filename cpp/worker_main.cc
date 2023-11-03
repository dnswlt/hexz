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

#include "base.h"
#include "board.h"
#include "hexz.pb.h"
#include "mcts.h"
#include "perfm.h"
#include "rpc.h"

namespace hexz {

using google::protobuf::RepeatedPtrFieldBackInserter;
using hexzpb::TrainingExample;

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
    NeuralMCTS mcts{model};
    Board b = Board::RandomBoard();
    mcts.PlayGame(b, config.runs_per_move, config.max_moves_per_game);
  }
}

void GenerateExamples(const Config& config) {
  const auto started_micros = UnixMicros();

  if (config.training_server_url != "") {
    RPCClient rpc(config);
    auto km = rpc.FetchLatestModel();
    if (!km.ok()) {
      ABSL_LOG(ERROR) << "FetchModule failed: " << km.status();
      return;
    }
    ABSL_LOG(INFO) << "Successully initialized model. Playing a game for fun.";
    NeuralMCTS mcts{km->model};
    Board b = Board::RandomBoard();
    auto examples =
        mcts.PlayGame(b, config.runs_per_move, config.max_moves_per_game);
    auto resp = rpc.SendExamples(km->key, examples);
    if (!resp.ok()) {
      ABSL_LOG(ERROR) << "Server did not like our examples: " << resp.status();
      return;
    }
    ABSL_LOG(INFO) << "Sent examples to training server at "
                   << config.training_server_url
                   << ". Response was: " << resp->DebugString();
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
  const char* training_server_url = std::getenv("HEXZ_TRAINING_SERVER_URL");
  auto config = hexz::Config{
      .test_url = hexz::GetEnv("HEXZ_TEST_URL"),
      .training_server_url = hexz::GetEnv("HEXZ_TRAINING_SERVER_URL"),
      .local_model_path = hexz::GetEnv("HEXZ_LOCAL_MODEL_PATH"),
      .runs_per_move = hexz::GetEnvAsInt("HEXZ_RUNS_PER_MOVE", 800),
      .max_moves_per_game = hexz::GetEnvAsInt("HEXZ_MAX_MOVES_PER_GAME", 200),
  };
  // Execute
  ABSL_LOG(INFO) << "Worker started with " << config.String();
  if (config.local_model_path != "") {
    ABSL_LOG(INFO) << "HEXZ_LOCAL_MODEL_PATH is set. Playing a game with a local model.";
    PlayGameLocally(config);
    return 0;
  }
  if (config.training_server_url != "") {
    ABSL_LOG(INFO) << "Generating examples and sending them to the training server.";
    hexz::GenerateExamples(config);
    return 0;
  }
  return 0;
}
