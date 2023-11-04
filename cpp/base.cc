#include "base.h"

#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <chrono>

namespace hexz {

namespace internal {
thread_local std::mt19937 rng{std::random_device{}()};
}  // namespace internal

std::string Config::String() const {
  return absl::StrCat(
      "Config(",
      absl::StrJoin(
          {
              absl::StrFormat("training_server_url: '%s'", training_server_url),
              absl::StrFormat("local_model_path: '%s'", local_model_path),
              absl::StrFormat("runs_per_move: %d", runs_per_move),
              absl::StrFormat("runs_per_move_gradient: %.3f",
                              runs_per_move_gradient),
              absl::StrFormat("max_moves_per_game: %d", max_moves_per_game),
              absl::StrFormat("max_runtime_seconds: %d", max_runtime_seconds),
              absl::StrFormat("max_games: %d", max_games),
              absl::StrFormat("uct_c: %.3f", uct_c),
          },
          ", "),
      ")");
}

Config Config::FromEnv() {
  return Config{
      .training_server_url = GetEnv("HEXZ_TRAINING_SERVER_URL"),
      .local_model_path = GetEnv("HEXZ_LOCAL_MODEL_PATH"),
      .runs_per_move = GetEnvAsInt("HEXZ_RUNS_PER_MOVE", 800),
      .runs_per_move_gradient =
          GetEnvAsDouble("HEXZ_RUNS_PER_MOVE_GRADIENT", -0.01),
      .max_moves_per_game = GetEnvAsInt("HEXZ_MAX_MOVES_PER_GAME", 200),
      .max_runtime_seconds = GetEnvAsInt("HEXZ_MAX_RUNTIME_SECONDS", 60),
      .max_games = GetEnvAsInt("HEXZ_MAX_GAMES", -1),
      .uct_c = static_cast<float>(GetEnvAsDouble("HEXZ_UCT_C", 5.0)),
  };
}

std::string GetEnv(const std::string& name) {
  const char* value = std::getenv(name.c_str());
  if (value == nullptr) {
    return "";
  }
  return std::string(value);
}

int GetEnvAsInt(const std::string& name, int default_value) {
  const char* value = std::getenv(name.c_str());
  if (value == nullptr) {
    return default_value;
  }
  return std::atoi(value);
}

double GetEnvAsDouble(const std::string& name, double default_value) {
  const char* value = std::getenv(name.c_str());
  if (value == nullptr) {
    return default_value;
  }
  char* end{};
  double d = std::strtod(value, &end);
  if (end == value) {
    // Could not parse float. Behave as GetEnvAsInt does.
    return 0;
  }
  return d;
}

int64_t UnixMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

}  // namespace hexz
