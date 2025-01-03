#include "base.h"

#include <absl/log/absl_check.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <array>
#include <cctype>
#include <chrono>
#include <random>
#include <sstream>
#include <thread>

namespace hexz {

namespace internal {
thread_local std::mt19937 rng{std::random_device{}()};
}  // namespace internal

std::string Config::String() const {
  return absl::StrCat(
      "Config(",
      absl::StrJoin(
          {
              absl::StrFormat("training_server_addr: '%s'",
                              training_server_addr),
              absl::StrFormat("device: '%s'", device),
              absl::StrFormat("worker_threads: %d", worker_threads),
              absl::StrFormat("fibers_per_thread: %d", fibers_per_thread),
              absl::StrFormat("prediction_batch_size: %d",
                              prediction_batch_size),
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
              absl::StrFormat("pin_threads: %d", pin_threads),
              absl::StrFormat("debug_memory_usage: %d", debug_memory_usage),
              absl::StrFormat("enable_health_service: %d",
                              enable_health_service),
              absl::StrFormat("suspend_while_training: %d",
                              suspend_while_training),
              absl::StrFormat("dry_run: %d", dry_run),
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
#define GET_ENV_BOOL(fld) \
  .fld = GetEnvAsBool(str_to_upper("HEXZ_" #fld), defaults.fld)

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
      GET_ENV_BOOL(pin_threads),
      GET_ENV_BOOL(debug_memory_usage),
      GET_ENV_BOOL(enable_health_service),
      GET_ENV_BOOL(suspend_while_training),
      GET_ENV_BOOL(dry_run),
  };
  // Special case: HEXZ_WORKER_SPEC is four values in one (to reduce clutter in
  // cloud env configs). Example: "cuda@4:128:256".
  std::string worker_spec = GetEnv("HEXZ_WORKER_SPEC", "");
  if (!worker_spec.empty()) {
    char device[16];
    if (std::sscanf(worker_spec.c_str(), "%15[^@]@%d:%d:%d", device,
                    &config.worker_threads, &config.fibers_per_thread,
                    &config.prediction_batch_size) != 4) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid HEXZ_WORKER_SPEC: %s", worker_spec));
    }
    config.device = std::string(device);
  }

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
#undef GET_ENV_BOOL
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

bool GetEnvAsBool(const std::string& name, bool default_value) {
  const char* value = std::getenv(name.c_str());
  if (value == nullptr) {
    return default_value;
  }
  std::string s(value);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s == "true" || s == "1" || s == "yes" || s == "on";
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

std::string RandomUid() {
  std::uniform_int_distribution<int> dis{0, 15};
  std::ostringstream os;
  os << std::hex;
  for (int i = 0; i < 8; i++) {
    os << dis(internal::rng);
  }
  return os.str();
}

}  // namespace hexz
