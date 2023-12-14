#include <absl/base/log_severity.h>
#include <absl/cleanup/cleanup.h>
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
#include "board.h"
#include "hexz.pb.h"
#include "mcts.h"
#include "perfm.h"
#include "rpc.h"

namespace hexz {

namespace {
std::string ModelId(const hexzpb::ModelKey& key) {
  return absl::StrCat(key.name(), ":", key.checkpoint());
}

bool SameKey(const hexzpb::ModelKey& lhs, const hexzpb::ModelKey& rhs) {
  return lhs.name() == rhs.name() && lhs.checkpoint() == rhs.checkpoint();
}

bool SameOrNewer(const hexzpb::ModelKey& lhs, const hexzpb::ModelKey& rhs) {
  return lhs.name() == rhs.name() && lhs.checkpoint() <= rhs.checkpoint();
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

}  // namespace

absl::StatusOr<std::unique_ptr<TorchModel>> FetchLatestModel(RPCClient& rpc) {
  auto km = rpc.FetchLatestModel();
  if (!km.ok()) {
    return km.status();
  }
  km->model.to(torch::kCPU);
  km->model.eval();
  return std::make_unique<TorchModel>(km->key, km->model);
}

void GenerateExamplesMultiThreaded(const Config& config) {
  const auto started_micros = UnixMicros();
  const std::string execution_id = RandomUid();
  ABSL_LOG(INFO) << "Generating examples using execution_id " << execution_id
                 << " and training server URL " << config.training_server_url;

  RPCClient rpc(config);
  auto keyed_model = rpc.FetchLatestModel();
  if (!keyed_model.ok()) {
    ABSL_LOG(ERROR) << "Failed to fetch latest model: " << keyed_model.status();
    return;
  }
  torch::DeviceType device = torch::kCPU;
  if (config.device == "mps") {
    device = torch::kMPS;
  } else if (config.device == "cuda") {
    device = torch::kCUDA;
  }
  // Never let a thread wait for its prediction to run for more than a 1s.
  constexpr int64_t timeout_micros = 1'000'000;
  BatchedTorchModel model(keyed_model->key, keyed_model->model, device,
                          config.worker_threads, timeout_micros);
  // Mutex that protects updates to the model
  std::mutex model_update_mut;

  std::vector<std::thread> worker_threads;
  for (int i = 0; i < config.worker_threads; i++) {
    worker_threads.emplace_back([&, thread_num = i] {
      auto token = model.RegisterThread();
      // Delay startup if requested.
      if (config.startup_delay_seconds > 0) {
        float delay = config.startup_delay_seconds * internal::UnitRandom();
        ABSL_LOG(INFO) << "Delaying startup by " << delay << " seconds.";
        std::this_thread::sleep_for(std::chrono::duration<float>(delay));
      }

      int max_games = config.max_games > 0 ? config.max_games
                                           : std::numeric_limits<int>::max();
      for (int i = 0; i < max_games; i++) {
        auto now = UnixMicros();
        if (now >=
            started_micros +
                static_cast<int64_t>(config.max_runtime_seconds) * 1'000'000) {
          ABSL_LOG(INFO) << "Time is up, aborting. " << now << " "
                         << config.max_runtime_seconds * 1'000'000;
          break;
        }
        NeuralMCTS mcts{model, std::make_unique<RandomPlayoutRunner>(), config};
        Board b = Board::RandomBoard();
        int64_t max_runtime_seconds =
            config.max_runtime_seconds - (now - started_micros) / 1'000'000;

        auto examples = mcts.PlayGame(b, max_runtime_seconds);

        if (!examples.ok()) {
          if (absl::IsDeadlineExceeded(examples.status())) {
            break;
          }
          ABSL_LOG(ERROR) << "Aborting: PlayGame returned an error: "
                          << examples.status();
          return;
        }
        const int n_examples = examples->size();
        ABSL_CHECK(n_examples > 0)
            << "Played a game that yielded no examples?!";
        const std::string model_id = ModelId(examples->back().model_key());

        auto resp = rpc.SendExamples(execution_id, *std::move(examples));

        if (!resp.ok()) {
          ABSL_LOG(ERROR) << "Failed to send examples: " << resp.status();
          return;
        }
        if (resp->status() ==
            hexzpb::AddTrainingExamplesResponse::REJECTED_AT_CAPACITY) {
          // For now, immediately exit if server is at capacity.
          // There are probably too many workers sending examples.
          ABSL_LOG(ERROR) << "Server is at capacity: " << resp->error_message();
          return;
        }
        if (resp->status() ==
            hexzpb::AddTrainingExamplesResponse::REJECTED_WRONG_MODEL) {
          // Since the server should accept both the previous and the latest
          // model version, this should be a rare event, and we should probably
          // tune our batch size. Individual workers are not able to generate
          // new examples before the model got updated twice.
          ABSL_LOG(ERROR)
              << "Server rejected out examples due to an old model. Sent: "
              << model_id << ", want: " << ModelId(resp->latest_model());
        } else if (resp->status() !=
                   hexzpb::AddTrainingExamplesResponse::ACCEPTED) {
          ABSL_LOG(ERROR) << "Server did not like our examples: "
                          << hexzpb::AddTrainingExamplesResponse::Status_Name(
                                 resp->status())
                          << ": " << resp->error_message();
          return;
        } else {
          ABSL_LOG(INFO) << "Successfully sent " << n_examples
                         << " examples to training server at "
                         << config.training_server_url;
        }
        // Check if we need to update to a new model.
        {
          std::scoped_lock<std::mutex> lk(model_update_mut);
          if (!SameOrNewer(resp->latest_model(), model.Key())) {
            auto keyed_model = rpc.FetchLatestModel();
            if (!keyed_model.ok()) {
              ABSL_LOG(ERROR)
                  << "Failed to fetch latest model: " << keyed_model.status();
              return;
            }
            model.UpdateModel(keyed_model->key, keyed_model->model);
            ABSL_LOG(INFO) << "Updated model to " << ModelId(keyed_model->key);
          }
        }
      }
      ABSL_LOG(INFO) << "Thread #" << thread_num << "("
                     << std::this_thread::get_id() << ") is done.";
    });
  }
  for (auto& t : worker_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void GenerateExamplesSingleThreaded(const Config& config) {
  if (config.startup_delay_seconds > 0) {
    float delay = config.startup_delay_seconds * internal::UnitRandom();
    ABSL_LOG(INFO) << "Delaying startup by " << delay << " seconds.";
    std::this_thread::sleep_for(std::chrono::duration<float>(delay));
  }
  const auto started_micros = UnixMicros();
  const std::string execution_id = RandomUid();
  ABSL_LOG(INFO) << "Generating examples using execution_id " << execution_id
                 << " and training server URL " << config.training_server_url;
  RPCClient rpc(config);
  auto model_or = FetchLatestModel(rpc);
  if (!model_or.ok()) {
    ABSL_LOG(ERROR) << "FetchModule failed: " << model_or.status();
    return;
  }
  std::unique_ptr<TorchModel> model = *std::move(model_or);
  if (config.device == "mps") {
    model->SetDevice(torch::kMPS);
  } else if (config.device == "cuda") {
    model->SetDevice(torch::kCUDA);
  }
  int max_games =
      config.max_games > 0 ? config.max_games : std::numeric_limits<int>::max();
  for (int i = 0; i < max_games; i++) {
    auto now = UnixMicros();
    if (now >=
        started_micros +
            static_cast<int64_t>(config.max_runtime_seconds) * 1'000'000) {
      ABSL_LOG(INFO) << "Time is up, aborting. " << now << " "
                     << config.max_runtime_seconds * 1'000'000;
      break;
    }
    NeuralMCTS mcts{*model, std::make_unique<RandomPlayoutRunner>(), config};
    Board b = Board::RandomBoard();
    int64_t max_runtime_seconds =
        config.max_runtime_seconds - (now - started_micros) / 1'000'000;

    auto examples = mcts.PlayGame(b, max_runtime_seconds);

    if (!examples.ok()) {
      if (absl::IsDeadlineExceeded(examples.status())) {
        break;
      }
      ABSL_LOG(ERROR) << "Aborting: PlayGame returned an error: "
                      << examples.status();
      return;
    }
    const int n_examples = examples->size();
    auto resp = rpc.SendExamples(execution_id, *std::move(examples));
    if (!resp.ok()) {
      ABSL_LOG(ERROR) << "Failed to send examples: " << resp.status();
      return;
    }
    if (resp->status() ==
        hexzpb::AddTrainingExamplesResponse::REJECTED_AT_CAPACITY) {
      // For now, immediately exit if server is at capacity.
      // There are too many workers sending examples.
      ABSL_LOG(ERROR) << "Server is at capacity: " << resp->error_message();
      return;
    }
    if (resp->status() ==
        hexzpb::AddTrainingExamplesResponse::REJECTED_WRONG_MODEL) {
      // Since the server should accept both the previous and the latest model
      // version, this should be a rare event, and we should probably tune our
      // batch size. Individual workers are not able to generate new examples
      // before the model got updated twice.
      ABSL_LOG(ERROR)
          << "Server rejected out examples due to an old model. Sent: "
          << ModelId(model->Key())
          << ", want: " << ModelId(resp->latest_model());
    }
    if (resp->status() != hexzpb::AddTrainingExamplesResponse::ACCEPTED) {
      ABSL_LOG(ERROR) << "Server did not like our examples: "
                      << hexzpb::AddTrainingExamplesResponse::Status_Name(
                             resp->status())
                      << ": " << resp->error_message();
      return;
    }
    // Check if we need to update to a new model.
    if (!SameKey(model->Key(), resp->latest_model())) {
      auto model_or = FetchLatestModel(rpc);
      if (!model_or.ok()) {
        ABSL_LOG(ERROR) << "FetchModule failed: " << model_or.status();
        return;
      }
      model = *std::move(model_or);
    }
    ABSL_LOG(INFO) << "Successfully sent " << n_examples
                   << " examples to training server at "
                   << config.training_server_url;
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

}  // namespace

int main() {
  // Initialization
  hexz::Perfm::InitScope perfm;
  // absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  // Config
  auto config = hexz::Config::FromEnv();
  if (!config.ok()) {
    ABSL_LOG(ERROR) << "Invalid config: " << config.status();
    return 1;
  }
  if (config->training_server_url.empty()) {
    ABSL_LOG(ERROR) << "training_server_url must be set";
    return 1;
  }
  hexz::InitializeFromConfig(*config);

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

  if (config->worker_threads > 1) {
    hexz::GenerateExamplesMultiThreaded(*config);
  } else {
    hexz::GenerateExamplesSingleThreaded(*config);
  }

  return 0;
}
