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

namespace hexz {

class GPUBenchmarkFn {
 public:
  struct input_t {
    torch::Tensor board;
    torch::Tensor action_mask;
  };
  class ComputeT {
   public:
    using input_t = GPUBenchmarkFn::input_t;
    using result_t = Model::Prediction;
    ComputeT(torch::jit::Module module, torch::DeviceType device)
        : module_{module}, device_{device} {
      module_.to(device);
    }
    std::vector<result_t> ComputeAll(std::vector<input_t>&& inputs) {
      const size_t n_inputs = inputs.size();
      std::vector<torch::Tensor> boards;
      boards.reserve(n_inputs);
      std::vector<torch::Tensor> action_masks;
      action_masks.reserve(n_inputs);
      for (auto& inp : inputs) {
        boards.emplace_back(std::move(inp.board));
        action_masks.emplace_back(std::move(inp.action_mask));
      }
      std::vector<torch::jit::IValue> model_inputs = {
          torch::stack(boards).to(device_),
          torch::stack(action_masks).to(device_),
      };
      auto pred = module_.forward(model_inputs);
      const auto output_tuple = pred.toTuple();
      // Read logits back to CPU / main memory.
      const auto probs =
          output_tuple->elements()[0].toTensor().exp().to(torch::kCPU);
      const auto values =
          output_tuple->elements()[1].toTensor().to(torch::kCPU);
      std::vector<Model::Prediction> result;
      result.reserve(n_inputs);
      for (int i = 0; i < n_inputs; i++) {
        result.push_back(Model::Prediction{
            .move_probs = probs.index({i}).reshape({2, 11, 10}),
            .value = values.index({i}).item<float>(),
        });
      }
      return result;
    }

   private:
    torch::jit::Module module_;
    torch::DeviceType device_;
  };

  GPUBenchmarkFn(torch::jit::Module module, torch::DeviceType device,
                 int batch_size)
      : batcher_(std::make_unique<ComputeT>(module, device), batch_size,
                 1'000'000) {}

  Model::Prediction Compute(ComputeT::input_t input) {
    return batcher_.ComputeValue(input);
  }

  ScopeGuard RegisterThread() { return batcher_.RegisterThread(); }

 private:
  Batcher<ComputeT> batcher_;
};

void RunGPUBenchmarkMultiThreaded(const Config& config) {
  ABSL_CHECK(config.worker_threads >= config.prediction_batch_size)
      << "worker_threads must be >= batch size";
  EmbeddedTrainingServiceClient client(absl::GetFlag(FLAGS_local_model_path));
  auto km = client.FetchLatestModel("");
  if (!km.ok()) {
    ABSL_LOG(ERROR) << "Failed to fetch model: " << km.status();
    return;
  }
  auto& [key, model] = *km;
  auto device = torch::kCPU;
  if (config.device == "mps") {
    device = torch::kMPS;
  } else if (config.device == "cuda") {
    device = torch::kCUDA;
  }
  GPUBenchmarkFn fn(model, device, config.prediction_batch_size);
  const int n_rounds = absl::GetFlag(FLAGS_gpu_benchmark_rounds);
  // Warmup
  {
    torch::NoGradGuard no_grad;
    auto guard = fn.RegisterThread();
    float sum = 0;
    torch::Tensor tb =
        torch::randn({11, 11, 10}, torch::dtype(torch::kFloat32));
    torch::Tensor ta = (torch::rand({2, 11, 10}) < 0.5);
    for (int j = 0; j < n_rounds / 10; j++) {
      auto result = fn.Compute(GPUBenchmarkFn::input_t{
          .board = tb,
          .action_mask = ta,
      });
    }
  }
  // Execution
  std::vector<std::thread> worker_threads;
  std::mutex mut;
  int op_count = 0;
  int64_t t_start = UnixMicros();
  for (int i = 0; i < config.worker_threads; i++) {
    worker_threads.emplace_back([&, thread_num = i] {
      torch::NoGradGuard no_grad;
      auto guard = fn.RegisterThread();
      float sum = 0;
      for (int j = 0; j < n_rounds; j++) {
        torch::Tensor tb =
            torch::randn({11, 11, 10}, torch::dtype(torch::kFloat32));
        torch::Tensor ta = (torch::rand({2, 11, 10}) < 0.5);
        auto result = fn.Compute(GPUBenchmarkFn::input_t{
            .board = tb,
            .action_mask = ta,
        });
        sum += result.value;
        {
          std::scoped_lock<std::mutex> l(mut);
          op_count++;
        }
      }
    });
  }
  for (auto& t : worker_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  int64_t t_end = UnixMicros();
  int final_op_count;
  {
    std::scoped_lock<std::mutex> l(mut);
    final_op_count = op_count;
  }
  float ops_per_sec =
      static_cast<float>(final_op_count) / (t_end - t_start) * 1e6;
  ABSL_LOG(INFO) << "Performed " << final_op_count << " computations in "
                 << (t_end - t_start) / 1000 << "ms (" << ops_per_sec
                 << " ops/s). batch size:" << config.prediction_batch_size
                 << ", threads:" << config.worker_threads;
}

void RunGPUBenchmark(const Config& config) {
  EmbeddedTrainingServiceClient client(absl::GetFlag(FLAGS_local_model_path));
  auto km = client.FetchLatestModel("");
  if (!km.ok()) {
    ABSL_LOG(ERROR) << "Failed to fetch model: " << km.status();
    return;
  }
  auto& [key, model] = *km;
  auto device = torch::kCPU;
  if (config.device == "mps") {
    device = torch::kMPS;
  } else if (config.device == "cuda") {
    device = torch::kCUDA;
  }
  model.to(device);
  torch::NoGradGuard no_grad;
  for (int batch_size = 1; batch_size <= 1024; batch_size *= 2) {
    int64_t t_start = 0;
    const int n_runs = 1000;
    float sum = 0;
    for (int i = 0; i < n_runs; i++) {
      if (i == n_runs / 10) {
        t_start = UnixMicros();
      }
      torch::Tensor tb =
          torch::randn({batch_size, 11, 11, 10}, torch::dtype(torch::kFloat32))
              .to(device);
      torch::Tensor ta =
          (torch::rand({batch_size, 2, 11, 10}) < 0.5).to(device);
      std::vector<torch::jit::IValue> inputs{tb, ta};
      auto output = model.forward(inputs);
      ABSL_CHECK(output.isTuple());
      const auto output_tuple = output.toTuple();
      ABSL_CHECK(output_tuple->size() == 2);
      const auto logits =
          output_tuple->elements()[0].toTensor().to(torch::kCPU);
      const auto dim = logits.sizes();
      ABSL_CHECK(dim.size() == 2 && dim[0] == batch_size &&
                 dim[1] == 2 * 11 * 10);
      const auto value =
          torch::sum(output_tuple->elements()[1].toTensor().to(torch::kCPU))
              .item<float>();
      sum += value;  // Just to avoid dead code pruning.
    }
    ABSL_DLOG(INFO) << "sum = " << sum;
    int64_t duration = UnixMicros() - t_start;
    int iterations = n_runs - n_runs / 10;
    ABSL_LOG(INFO) << "batch_size=" << batch_size << ": finished " << iterations
                   << " iterations in " << (duration / 1000) << "ms ("
                   << (static_cast<float>(iterations * batch_size) / duration *
                       1e6)
                   << " ops/s)";
  }
}

}  // namespace hexz

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
  if (config->training_server_url.empty()) {
    ABSL_LOG(ERROR) << "training_server_url must be set";
    return 1;
  }
  hexz::InitializeFromConfig(*config);

  if (std::string bench = absl::GetFlag(FLAGS_run_gpu_benchmark); bench != "") {
    // Run GPU performance test and exit.
    if (bench == "multi") {
      hexz::RunGPUBenchmarkMultiThreaded(*config);
    } else {
      hexz::RunGPUBenchmark(*config);
    }
    return 0;
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
