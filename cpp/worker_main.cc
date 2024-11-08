#include <absl/base/log_severity.h>
#include <absl/cleanup/cleanup.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/log/absl_log.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <utility>

#include "base.h"
#include "batch.h"
#include "board.h"
#include "grpc_client.h"
#include "hexz.pb.h"
#include "mcts.h"
#include "perfm.h"
#include "worker.h"

// In Docker, you should override conig defaults using HEXZ_* environment
// variables. For convenience, we add flag overrides as well for the most
// relevant configs for local testing.
ABSL_FLAG(std::string, training_server_addr, "",
          "training server address (ex: \"localhost:50051\")");
ABSL_FLAG(std::string, device, "", "PyTorch device (cpu, cuda, mps)");
ABSL_FLAG(std::string, local_model_path, "",
          "path to a torch::jit::load'able model (for benchmarks)");
ABSL_FLAG(int, runs_per_move, 0, "number of MCTS runs per move");
ABSL_FLAG(int, worker_threads, 0, "number of worker threads");
ABSL_FLAG(int, fibers_per_thread, 0, "number of fibers per worker thread");
ABSL_FLAG(int, prediction_batch_size, 0,
          "batch size for GPU model predictions");
ABSL_FLAG(int, max_runtime_seconds, 0, "maximum runtime of the worker");
ABSL_FLAG(int, max_games, 0, "maximum number of games to play");
ABSL_FLAG(bool, suspend_while_training, false,
          "if true, the worker is suspended during training");
ABSL_FLAG(bool, dry_run, false,
          "if true, the worker will not send examples to the training server");
ABSL_FLAG(bool, pin_threads, false,
          "if true, worker threads will be pinned to CPU cores.");

namespace {

struct BackgroundThreadSignal {
  std::condition_variable cv;
  std::mutex mut;
  bool stop = false;
};

void MemMon(BackgroundThreadSignal& sig) {
  std::unique_lock<std::mutex> lk(sig.mut);
  while (!sig.cv.wait_for(lk, std::chrono::duration<float>(5.0),
                          [&sig] { return sig.stop; })) {
    std::ifstream infile("/proc/self/status");
    if (!infile.is_open()) {
      ABSL_LOG(ERROR)
          << "Cannot open /proc/self/status. Terminating MemMon thread.";
      return;
    }
    std::string line;
    while (std::getline(infile, line)) {
      if (line.compare(0, 2, "Vm") == 0) {
        ABSL_LOG(INFO) << line;
      }
    }
  }
}

void APMMon(BackgroundThreadSignal& sig, float period) {
  std::unique_lock<std::mutex> lk(sig.mut);
  while (!sig.cv.wait_for(lk, std::chrono::duration<float>(period),
                          [&sig] { return sig.stop; })) {
    double examples_rate_1m = hexz::APMExamples().Rate(60);
    double games_rate_1m = hexz::APMGames().Rate(60);
    double examples_rate_10s = hexz::APMExamples().Rate(10);
    double games_rate_10s = hexz::APMGames().Rate(10);
    ABSL_LOG(INFO) << "APM(1m): examples/s: " << examples_rate_1m
                   << " games/s: " << games_rate_1m
                   << " -- APM(10s): examples/s: " << examples_rate_10s
                   << " games/s: " << games_rate_10s;
  }
}

void UpdateConfigFromFlags(hexz::Config& config) {
  if (std::string addr = absl::GetFlag(FLAGS_training_server_addr);
      addr != "") {
    config.training_server_addr = addr;
  }
  if (std::string device = absl::GetFlag(FLAGS_device); device != "") {
    config.device = device;
  }
  if (int n = absl::GetFlag(FLAGS_runs_per_move); n > 0) {
    config.runs_per_move = n;
  }
  if (int t = absl::GetFlag(FLAGS_worker_threads); t > 0) {
    config.worker_threads = t;
  }
  if (int t = absl::GetFlag(FLAGS_fibers_per_thread); t > 0) {
    config.fibers_per_thread = t;
  }
  if (int s = absl::GetFlag(FLAGS_prediction_batch_size); s > 0) {
    config.prediction_batch_size = s;
  }
  if (int t = absl::GetFlag(FLAGS_max_runtime_seconds); t > 0) {
    config.max_runtime_seconds = t;
  }
  if (int n = absl::GetFlag(FLAGS_max_games); n > 0) {
    config.max_games = n;
  }
  if (bool b = absl::GetFlag(FLAGS_suspend_while_training); b) {
    config.suspend_while_training = b;
  }
  if (bool b = absl::GetFlag(FLAGS_dry_run); b) {
    config.dry_run = b;
  }
  if (bool b = absl::GetFlag(FLAGS_pin_threads); b) {
    config.pin_threads = b;
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  // Initialization
  // absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
  absl::SetProgramUsageMessage(
      "Generate training examples."
      "\nConfigure using HEXZ_* environment variables, optionally overriding "
      "them with these flags:");
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  hexz::Perfm::InitScope perfm;
  // Config
  auto config = hexz::Config::FromEnv();
  if (!config.ok()) {
    ABSL_LOG(ERROR) << "Invalid config: " << config.status();
    return 1;
  }
  UpdateConfigFromFlags(*config);
  hexz::InitializeFromConfig(*config);
  // Cannot run without a training server URL, so exit early if it is not set.
  if (config->training_server_addr.empty()) {
    ABSL_LOG(ERROR) << "training_server_addr must be set";
    return 1;
  }
  ABSL_LOG(INFO) << "Worker started with " << config->String();

  // Memory usage monitoring, if requested.
  std::thread memmon;
  BackgroundThreadSignal memmon_sig;
  absl::Cleanup thread_joiner = [&memmon, &memmon_sig] {
    if (memmon.joinable()) {
      ABSL_LOG(INFO) << "Joining memory monitoring thread.";
      {
        std::lock_guard<std::mutex> lk(memmon_sig.mut);
        memmon_sig.stop = true;
      }
      memmon_sig.cv.notify_all();
      memmon.join();
    }
  };
  if (config->debug_memory_usage) {
    memmon = std::thread{MemMon, std::ref(memmon_sig)};
  }
  // Application performance monitoring.
  std::thread apmmon;
  BackgroundThreadSignal apmmon_sig;
  absl::Cleanup apm_thread_joiner = [&apmmon, &apmmon_sig] {
    if (apmmon.joinable()) {
      ABSL_LOG(INFO) << "Joining performance monitoring thread.";
      {
        std::lock_guard<std::mutex> lk(apmmon_sig.mut);
        apmmon_sig.stop = true;
      }
      apmmon_sig.cv.notify_all();
      apmmon.join();
    }
  };
  apmmon = std::thread{APMMon, std::ref(apmmon_sig), /*period=*/5.0};

  // Execute
  std::unique_ptr<hexz::TrainingServiceClient> client;
  if (std::string path = absl::GetFlag(FLAGS_local_model_path); path != "") {
    ABSL_LOG(INFO) << "Using EmbeddedTrainingServiceClient(" << path << ")";
    client = std::make_unique<hexz::EmbeddedTrainingServiceClient>(path);
  } else {
    client =
        hexz::GRPCTrainingServiceClient::Connect(config->training_server_addr);
  }

  hexz::Worker worker(*config, *client);
  worker.Run();

  return 0;
}
