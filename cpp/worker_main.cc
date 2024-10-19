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
#include <sstream>
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


ABSL_FLAG(std::string, training_server_addr, "",
          "training server address (ex: \"localhost:50051\")");
ABSL_FLAG(std::string, device, "", "PyTorch device (cpu, cuda, mps)");
ABSL_FLAG(std::string, run_gpu_benchmark, "",
          "set to \"single\" or \"multi\" a GPU benchmark");
ABSL_FLAG(int, gpu_benchmark_rounds, 0,
          "number of rounds to run in a GPU benchmark");
ABSL_FLAG(std::string, local_model_path, "",
          "path to a torch::jit::load'able model (for benchmarks)");
ABSL_FLAG(int, worker_threads, 0, "number of worker threads");
ABSL_FLAG(int, prediction_batch_size, 0,
          "batch size for GPU model predictions");
ABSL_FLAG(int, max_runtime_seconds, 0, "maximum runtime of the worker");

namespace {
std::condition_variable cv_memmon;
std::mutex cv_memmon_mut;
bool stop_memmon = false;

void MemMon() {
  std::unique_lock<std::mutex> lk(cv_memmon_mut);
  while (!cv_memmon.wait_for(lk, std::chrono::duration<float>(5.0),
                             [] { return stop_memmon; })) {
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

void UpdateConfigFromFlags(hexz::Config& config) {
  if (std::string addr = absl::GetFlag(FLAGS_training_server_addr);
      addr != "") {
    config.training_server_url = addr;
  }
  if (std::string device = absl::GetFlag(FLAGS_device); device != "") {
    config.device = device;
  }
  if (int t = absl::GetFlag(FLAGS_worker_threads); t > 0) {
    config.worker_threads = t;
  }
  if (int s = absl::GetFlag(FLAGS_prediction_batch_size); s > 0) {
    config.prediction_batch_size = s;
  }
  if (int t = absl::GetFlag(FLAGS_max_runtime_seconds); t > 0) {
    config.max_runtime_seconds = t;
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
  if (config->training_server_url.empty()) {
    ABSL_LOG(ERROR) << "training_server_url must be set";
    return 1;
  }
  // Execute
  ABSL_LOG(INFO) << "Worker started with " << config->String();
  std::thread memmon;
  absl::Cleanup thread_joiner = [&memmon] {
    if (memmon.joinable()) {
      ABSL_LOG(INFO) << "Joining memory monitoring thread.";
      {
        std::lock_guard<std::mutex> lk(cv_memmon_mut);
        stop_memmon = true;
      }
      cv_memmon.notify_one();
      memmon.join();
    }
  };
  if (config->debug_memory_usage) {
    memmon = std::thread{MemMon};
  }

  std::unique_ptr<hexz::TrainingServiceClient> client;
  if (std::string path = absl::GetFlag(FLAGS_local_model_path); path != "") {
    ABSL_LOG(INFO) << "Using EmbeddedTrainingServiceClient(" << path << ")";
    client = std::make_unique<hexz::EmbeddedTrainingServiceClient>(path);
  } else {
    client =
        hexz::GRPCTrainingServiceClient::Connect(config->training_server_url);
  }

  hexz::GenerateExamplesMultiThreaded(*config, *client);

  return 0;
}
