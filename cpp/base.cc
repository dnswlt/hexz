#include "base.h"

#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <cctype>
#include <chrono>
#include <random>
#include <thread>

namespace hexz {

namespace internal {
thread_local std::mt19937 rng{std::random_device{}()};

float UnitRandom() {
  std::uniform_real_distribution<float> unif(0, 1);
  return unif(rng);
}

std::vector<float> Dirichlet(int n, float concentration) {
  std::gamma_distribution<float> gamma;
  std::vector<float> v(n);
  float sum = 0;
  for (int i = 0; i < n; i++) {
    v[i] = gamma(rng);
    sum += v[i];
  }
  for (int i = 0; i < n; i++) {
    v[i] /= sum;
  }
  return v;
}

void RandomDelay(float max_delay_seconds) {
  if (max_delay_seconds <= 0) {
    return;
  }
  std::uniform_real_distribution<float> dis(0, max_delay_seconds);
  float delay = dis(rng);
  std::this_thread::sleep_for(std::chrono::duration<float>(delay));
}

}  // namespace internal

std::string Config::String() const {
  return absl::StrCat(
      "Config(",
      absl::StrJoin(
          {
              absl::StrFormat("training_server_url: '%s'", training_server_url),
              absl::StrFormat("local_model_path: '%s'", local_model_path),
              absl::StrFormat("runs_per_move: %d", runs_per_move),
              absl::StrFormat("runs_per_fast_move: %d", runs_per_fast_move),
              absl::StrFormat("fast_move_prob: %.3f", fast_move_prob),
              absl::StrFormat("runs_per_move_gradient: %.3f",
                              runs_per_move_gradient),
              absl::StrFormat("max_moves_per_game: %d", max_moves_per_game),
              absl::StrFormat("max_runtime_seconds: %d", max_runtime_seconds),
              absl::StrFormat("max_games: %d", max_games),
              absl::StrFormat("uct_c: %.3f", uct_c),
              absl::StrFormat("dirichlet_concentration: %.3f",
                              dirichlet_concentration),
              absl::StrFormat("startup_delay_seconds: %.3f",
                              startup_delay_seconds),
          },
          ", "),
      ")");
}

namespace {
std::string str_to_upper(const std::string& s) {
  std::string t = s;
  std::transform(s.begin(), s.end(), t.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return t;
}
}  // namespace

#define GET_ENV(fld) .fld = GetEnv(str_to_upper("HEXZ_" #fld), defaults.fld)
#define GET_ENV_INT(fld) \
  .fld = GetEnvAsInt(str_to_upper("HEXZ_" #fld), defaults.fld)
#define GET_ENV_FLOAT(fld) \
  .fld = GetEnvAsFloat(str_to_upper("HEXZ_" #fld), defaults.fld)

Config Config::FromEnv() {
  Config defaults{};
  return Config{
      GET_ENV(training_server_url),
      GET_ENV(local_model_path),
      GET_ENV_INT(runs_per_move),
      GET_ENV_INT(runs_per_fast_move),
      GET_ENV_FLOAT(fast_move_prob),
      GET_ENV_FLOAT(runs_per_move_gradient),
      GET_ENV_INT(max_moves_per_game),
      GET_ENV_INT(max_runtime_seconds),
      GET_ENV_INT(max_games),
      GET_ENV_FLOAT(uct_c),
      GET_ENV_FLOAT(dirichlet_concentration),
      GET_ENV_FLOAT(startup_delay_seconds),
  };
}

#undef GET_ENV
#undef GET_ENV_INT
#undef GET_ENV_DOUBLE

std::string GetEnv(const std::string& name, const std::string& default_value) {
  const char* value = std::getenv(name.c_str());
  if (value == nullptr) {
    return default_value;
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

float GetEnvAsFloat(const std::string& name, float default_value) {
  const char* value = std::getenv(name.c_str());
  if (value == nullptr) {
    return default_value;
  }
  char* end{};
  float d = static_cast<float>(std::strtod(value, &end));
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
