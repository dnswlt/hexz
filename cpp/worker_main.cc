#include <cpr/cpr.h>
#include <torch/script.h>

#include <algorithm>
#include <cassert>
#include <ctime>
#include <iostream>

#include "board.h"
#include "hexz.pb.h"
#include "mcts.h"
#include "util.h"

namespace hexz {

using hexzpb::TrainingExample;

struct Config {
  std::string test_url;
  std::string training_server_url;
  // Path to a local model.pt file.
  // Can be used for local runs without the training server.
  std::string local_model_path;
  int runs_per_move;
};

void PlayGameLocally(const Config& config) {
  torch::jit::script::Module model;
  try {
    model = torch::jit::load(config.local_model_path);
    model.to(torch::kCPU);
    model.eval();
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load model from " << config.local_model_path << ": "
              << e.msg() << "\n";
    return;
  }
  {
    Perfm::Scope ps(Perfm::PlayGameLocally);
    NeuralMCTS mcts{model};
    Board b = Board::RandomBoard();
    mcts.PlayGame(b, config.runs_per_move, /*max_moves=*/3);
  }
}

void TrialRun(const Config& config) {
  const auto started_micros = UnixMicros();

  // Check that cpr works:
  if (config.test_url != "") {
    cpr::Response r = cpr::Get(
        cpr::Url{"https://api.github.com/repos/whoshuu/cpr/contributors"},
        cpr::Authentication{"user", "pass", cpr::AuthMode::BASIC},
        cpr::Parameters{{"anon", "true"}, {"key", "value"}});
    std::cout << "Status: " << r.status_code << "\n";
    std::cout << "content-type: " << r.header["content-type"] << "\n";
    std::cout << r.text << "\n";
  }

  // Check that torch works:
  Board b = Board::RandomBoard();
  assert(b.Flags(0) == 3);
  assert(b.Flags(1) == 3);
  auto zero_score = std::make_pair(0.0f, 0.0f);
  assert(b.Score() == zero_score);
  int player = 0;
  for (int i = 0; i < 10; i++) {
    auto moves = b.NextMoves(player);
    assert(moves.size() > 0);
    b.MakeMove(player, moves[0]);
    player = 1 - player;
  }
  assert(b.Score() != zero_score);
  std::cout << b.Score() << "\n";

  if (config.local_model_path != "") {
    PlayGameLocally(config);
  } else {
    std::cout << "HEXZ_LOCAL_MODEL_PATH not set. Skipping game play.\n";
  }

  // Check that protobuf works:
  const int64_t duration_micros = UnixMicros() - started_micros;
  TrainingExample example;
  example.set_result(1.0);
  example.set_duration_micros(duration_micros);
  example.set_unix_micros(started_micros);
  std::cout << "Hello, hexz: " << example.DebugString() << "\n";
}

}  // namespace hexz

int main() {
  hexz::Perfm::Init();
  const char* training_server_url = std::getenv("HEXZ_TRAINING_SERVER_URL");
  auto config = hexz::Config{
      .test_url = hexz::GetEnv("HEXZ_TEST_URL"),
      .training_server_url = hexz::GetEnv("HEXZ_TRAINING_SERVER_URL"),
      .local_model_path = hexz::GetEnv("HEXZ_LOCAL_MODEL_PATH"),
      .runs_per_move = hexz::GetEnvAsInt("HEXZ_RUNS_PER_MOVE", 800),
  };
  hexz::TrialRun(config);
  hexz::Perfm::PrintStats();
  return 0;
}
