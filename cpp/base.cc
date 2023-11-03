#include "base.h"

#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <chrono>

namespace hexz {

std::string Config::String() const {
  return absl::StrCat(
      "Config(",
      absl::StrJoin(
          {
              absl::StrFormat("test_url: '%s'", test_url),
              absl::StrFormat("training_server_url: '%s'", training_server_url),
              absl::StrFormat("local_model_path: '%s'", local_model_path),
              absl::StrFormat("runs_per_move: %d", runs_per_move),
              absl::StrFormat("max_moves_per_game: %d", max_moves_per_game),
              absl::StrFormat("max_runtime_seconds: %d", max_runtime_seconds),
          },
          ", "),
      ")");
}

Config Config::FromEnv() {
  return Config{
      .test_url = GetEnv("HEXZ_TEST_URL"),
      .training_server_url = GetEnv("HEXZ_TRAINING_SERVER_URL"),
      .local_model_path = GetEnv("HEXZ_LOCAL_MODEL_PATH"),
      .runs_per_move = GetEnvAsInt("HEXZ_RUNS_PER_MOVE", 800),
      .max_moves_per_game = GetEnvAsInt("HEXZ_MAX_MOVES_PER_GAME", 200),
      .max_runtime_seconds = GetEnvAsInt("HEXZ_MAX_RUNTIME_SECONDS", 60),
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

int64_t UnixMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

}  // namespace hexz
