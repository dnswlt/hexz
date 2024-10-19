#include "base.h"

#include <absl/log/absl_check.h>
#include <absl/status/statusor.h>
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

int RandomInt(int lower, int upper) {
  ABSL_DCHECK(lower <= upper);
  std::uniform_int_distribution<int> dis{lower, upper};
  return dis(rng);
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

}  // namespace internal

std::string Config::String() const {
  return absl::StrCat(
      "Config(",
      absl::StrJoin(
          {
              absl::StrFormat("training_server_addr: '%s'", training_server_addr),
              absl::StrFormat("device: '%s'", device),
              absl::StrFormat("worker_threads: %d", worker_threads),
              absl::StrFormat("fibers_per_thread: %d", fibers_per_thread),
              absl::StrFormat("prediction_batch_size: %d", prediction_batch_size),
              absl::StrFormat("runs_per_move: %d", runs_per_move),
              absl::StrFormat("runs_per_fast_move: %d", runs_per_fast_move),
              absl::StrFormat("fast_move_prob: %.3f", fast_move_prob),
              absl::StrFormat("max_moves_per_game: %d", max_moves_per_game),
              absl::StrFormat("max_runtime_seconds: %d", max_runtime_seconds),
              absl::StrFormat("max_games: %d", max_games),
              absl::StrFormat("uct_c: %.3f", uct_c),
              absl::StrFormat("initial_root_q_value: %.3f",
                              initial_root_q_value),
              absl::StrFormat("initial_q_penalty: %.3f", initial_q_penalty),
              absl::StrFormat("dirichlet_concentration: %.3f",
                              dirichlet_concentration),
              absl::StrFormat("random_playouts: %d", random_playouts),
              absl::StrFormat("resign_threshold: %.6f", resign_threshold),
              absl::StrFormat("startup_delay_seconds: %.3f",
                              startup_delay_seconds),
              absl::StrFormat("debug_memory_usage: %d", debug_memory_usage),
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


absl::StatusOr<Config> Config::FromEnv() {
  Config defaults{};

#define GET_ENV(fld) .fld = GetEnv(str_to_upper("HEXZ_" #fld), defaults.fld)
#define GET_ENV_INT(fld) \
  .fld = GetEnvAsInt(str_to_upper("HEXZ_" #fld), defaults.fld)
#define GET_ENV_FLOAT(fld) \
  .fld = GetEnvAsFloat(str_to_upper("HEXZ_" #fld), defaults.fld)

  Config config{
      GET_ENV(training_server_addr),
      GET_ENV(device),
      GET_ENV_INT(worker_threads),
      GET_ENV_INT(fibers_per_thread),
      GET_ENV_INT(prediction_batch_size),
      GET_ENV_INT(runs_per_move),
      GET_ENV_INT(runs_per_fast_move),
      GET_ENV_FLOAT(fast_move_prob),
      GET_ENV_INT(max_moves_per_game),
      GET_ENV_INT(max_runtime_seconds),
      GET_ENV_INT(max_games),
      GET_ENV_FLOAT(uct_c),
      GET_ENV_FLOAT(initial_root_q_value),
      GET_ENV_FLOAT(initial_q_penalty),
      GET_ENV_FLOAT(dirichlet_concentration),
      GET_ENV_INT(random_playouts),
      GET_ENV_FLOAT(resign_threshold),
      GET_ENV_FLOAT(startup_delay_seconds),
      GET_ENV_INT(debug_memory_usage),
  };
  // Validate values
  const std::vector<std::string> valid_devices = {"cpu", "mps", "cuda"};
  if (std::find(valid_devices.begin(), valid_devices.end(), config.device) ==
      valid_devices.end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid device: %s. Must be one of (cpu, mps, cuda).", config.device));
  }
  return config;

#undef GET_ENV
#undef GET_ENV_INT
#undef GET_ENV_DOUBLE

}


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
