#include <absl/base/log_severity.h>
#include <absl/log/absl_log.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <cpr/cpr.h>
#include <google/protobuf/util/json_util.h>
#include <torch/script.h>

#include <algorithm>
#include <cassert>
#include <ctime>
#include <iostream>
#include <sstream>

#include "board.h"
#include "hexz.pb.h"
#include "mcts.h"
#include "util.h"

namespace hexz {

using google::protobuf::RepeatedPtrFieldBackInserter;
using hexzpb::TrainingExample;

struct Config {
  std::string test_url;
  std::string training_server_url;
  // Path to a local model.pt file.
  // Can be used for local runs without the training server.
  std::string local_model_path;
  int runs_per_move;
  int max_moves_per_game;

  std::string String() const {
    return absl::StrCat(
        "Config(",
        absl::StrJoin(
            {
                absl::StrFormat("test_url: '%s'", test_url),
                absl::StrFormat("training_server_url: '%s'",
                                training_server_url),
                absl::StrFormat("local_model_path: '%s'", local_model_path),
                absl::StrFormat("runs_per_move: %d", runs_per_move),
                absl::StrFormat("max_moves_per_game: %d", max_moves_per_game),
            },
            ", "),
        ")");
  }
};

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

struct KeyedModel {
  hexzpb::ModelKey key;
  torch::jit::Module model;
};

absl::StatusOr<KeyedModel> FetchModule(const Config& config) {
  // Get info about the current model.
  cpr::Response key_resp = cpr::Get(
      cpr::Url{config.training_server_url + "/models/current"},
      cpr::Timeout{1000}, cpr::Header{{"Accept", "application/x-protobuf"}});
  if (key_resp.status_code == 0) {
    return absl::UnavailableError(
        absl::StrCat("Server unreachable: ", key_resp.error.message));
  }
  if (key_resp.status_code != 200) {
    return absl::AbortedError(
        absl::StrCat("Server returned status code ", key_resp.status_code));
  }
  hexzpb::ModelKey model_key;
  if (auto status = google::protobuf::util::JsonStringToMessage(
          key_resp.text, &model_key,
          google::protobuf::util::JsonParseOptions());
      !status.ok()) {
    return status;
  }
  ABSL_LOG(INFO) << "Fetching model " << model_key.name() << ":"
                 << model_key.checkpoint();
  cpr::Response model_resp =
      cpr::Get(cpr::Url{absl::StrCat(config.training_server_url, "/models/",
                                     model_key.name(), "/checkpoints/",
                                     model_key.checkpoint())},
               cpr::Parameters{{"repr", "scriptmodule"}},  //
               cpr::Timeout{1000});
  if (model_resp.status_code == 404) {
    // This *might* be a concurrent model change between our call to
    // /models/current and the failed fetch. We should try again.
    return absl::NotFoundError("Server returned 404");
  }
  if (model_resp.status_code != 200) {
    return absl::AbortedError(absl::StrCat(
        "Fetch model: server returned status code ", model_resp.status_code));
  }
  try {
    std::istringstream model_is(model_resp.text);
    auto model = torch::jit::load(model_is);  // , torch::kCPU);
    model.to(torch::kCPU);
    model.eval();
    return KeyedModel{
        .key = model_key,
        .model = model,
    };
  } catch (const c10::Error& e) {
    return absl::InternalError(
        absl::StrCat("Failed to load torch module: ", e.msg()));
  }
}

void TrialRun(const Config& config) {
  const auto started_micros = UnixMicros();

  if (config.local_model_path != "") {
    PlayGameLocally(config);
  } else {
    ABSL_LOG(INFO) << "HEXZ_LOCAL_MODEL_PATH not set. Skipping game play.";
  }

  if (config.training_server_url != "") {
    auto km = FetchModule(config);
    if (!km.ok()) {
      ABSL_LOG(ERROR) << "FetchModule failed: " << km.status();
      return;
    }
    ABSL_LOG(INFO) << "Successully initialized model. Playing a game for fun.";
    NeuralMCTS mcts{km->model};
    Board b = Board::RandomBoard();
    auto examples =
        mcts.PlayGame(b, config.runs_per_move, config.max_moves_per_game);
    hexzpb::AddTrainingExamplesRequest req;
    *req.mutable_model_key() = km->key;
    req.mutable_examples()->Reserve(examples.size());
    std::move(examples.begin(), examples.end(),
              RepeatedPtrFieldBackInserter(req.mutable_examples()));
    cpr::Response resp = cpr::Post(
        cpr::Url{config.training_server_url + "/examples"}, cpr::Timeout{1000},
        cpr::Header{{"Content-Type", "application/x-protobuf"}},
        cpr::Body{req.SerializeAsString()});
    if (resp.status_code != 200) {
      ABSL_LOG(ERROR) << "Server did not like our request: " <<resp.status_code;
      return;
    }
    hexzpb::AddTrainingExamplesResponse response;
    if (!response.ParseFromString(resp.text)) {
      ABSL_LOG(ERROR) << "Cannot parse reponse as AddTrainingExamplesResponse";
      return;
    }
    ABSL_LOG(INFO) << "Nice response from the server: " << response.DebugString();
  }
}

}  // namespace hexz

int main() {
  // Initialization
  hexz::Perfm::Init();
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
  hexz::TrialRun(config);
  // Tear down
  hexz::Perfm::PrintStats();
  return 0;
}
