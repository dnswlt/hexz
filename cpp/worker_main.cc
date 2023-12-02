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
#include <cassert>
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

void GenerateExamples(const Config& config) {
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
    auto resp =
        rpc.SendExamples(execution_id, model->Key(), *std::move(examples));
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

std::mutex m_out;

void SyncPrint(const std::string& s) {
  std::scoped_lock<std::mutex> l(m_out);
  std::cout << s;
}

class IntCompute {
 public:
  using input_t = int;
  using result_t = int;

  int AddInput(input_t v) {
    inputs_.push_back(v);
    return inputs_.size() - 1;
  }

  void ComputeAll() {
    for (const auto& inp : inputs_) {
      results_.push_back(ComputeOne(inp));
    }
    // Pretend the computation takes a while
    int64_t t_before = hexz::UnixMicros();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    SyncPrint(absl::StrFormat("Slept for %d us.\n", hexz::UnixMicros() - t_before));
  }

  result_t GetResult(int idx) {
    ABSL_CHECK(idx >= 0 && idx < results_.size());
    return results_[idx];
  }

  void Reset() {
    inputs_.clear();
    results_.clear();
  }

 private:
  result_t ComputeOne(const input_t& v) { return v * v; }

  std::vector<input_t> inputs_;
  std::vector<result_t> results_;
};

template <typename ComputeT>
class Batcher {
 public:
  using input_t = typename ComputeT::input_t;
  using result_t = typename ComputeT::result_t;

  Batcher(ComputeT comp, int batch_size)
      : comp_(comp), slots_(batch_size), batch_size_(batch_size) {}

  int ComputeValue(int thread_id, input_t v) {
    std::unique_lock<std::mutex> l(m_);
    SyncPrint(absl::StrFormat("Thread %d: wait for cv_enter\n", thread_id));
    cv_enter_.wait(l, [this] { return slots_ > 0; });
    SyncPrint(absl::StrFormat("Thread %d: cv_enter with slots_ == %d\n",
                              thread_id, slots_));
    slots_--;
    int result_key = comp_.AddInput(v);
    if (slots_ > 0) {
      // wait for next batch to be filled.
      SyncPrint(absl::StrFormat("Thread %d: wait for cv_ready\n", thread_id));
      cv_ready_.wait(l, [this] { return batch_ready_; });
      SyncPrint(absl::StrFormat("Thread %d: cv_ready\n", thread_id));
      done_++;
      result_t result = comp_.GetResult(result_key);
      if (done_ == batch_size_) {
        // This is the last thread leaving the batch.
        comp_.Reset();
        batch_ready_ = false;
        slots_ = batch_size_;
        done_ = 0;
        l.unlock();
        cv_enter_.notify_all();
      }
      return result;
    }
    // The thread that got the last slot in the batch has to
    // compute values for all waiting threads.
    comp_.ComputeAll();

    batch_ready_ = true;
    result_t result = comp_.GetResult(result_key);
    done_++;
    if (done_ == batch_size_) {
      // This is the last thread leaving the batch. Can only happen here
      // if batch size is 1.
      SyncPrint(
          absl::StrFormat("!!! Thread %d: notify cv_enter !!!\n", thread_id));
      comp_.Reset();
      batch_ready_ = false;
      slots_ = batch_size_;
      done_ = 0;
      l.unlock();
      cv_enter_.notify_all();
    } else {
      SyncPrint(absl::StrFormat("Thread %d: notify cv_ready\n", thread_id));
      l.unlock();              // avoid pessimization.
      cv_ready_.notify_all();  // notify others waiting on batch results.
    }
    return result;
  }

 private:
  ComputeT comp_;
  bool batch_ready_ = false;
  int batch_size_ = 0;
  int slots_ = 0;
  int done_ = 0;
  std::mutex m_;
  std::condition_variable cv_enter_;
  std::condition_variable cv_ready_;
};

void TRun(int id, int rounds, Batcher<IntCompute>& batcher) {
  for (int i = 0; i < rounds; i++) {
    int v = batcher.ComputeValue(id, i);
    SyncPrint(absl::StrFormat("Thread %d: f(%d) = %d\n", id, i, v));
  }
  SyncPrint(absl::StrFormat("Thread %d: DONE\n", id));
}

void RunMultithreadExperiment() {
  constexpr int n_threads = 8;
  constexpr int n_rounds = 100;
  std::vector<std::thread> ts;
  Batcher<IntCompute> b(IntCompute(), n_threads);
  int64_t t_started = hexz::UnixMicros();
  for (int i = 0; i < n_threads; i++) {
    ts.emplace_back([&b, i] { TRun(i, n_rounds, b); });
  }
  for (auto& t : ts) {
    if (t.joinable()) {
      t.join();
    }
  }
  int64_t t_done = hexz::UnixMicros();
  std::cout << "Done with " << n_rounds << " rounds in "
            << static_cast<double>(t_done - t_started) / 1e6 << " seconds\n";
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
  if (config.training_server_url.empty()) {
    ABSL_LOG(ERROR) << "training_server_url must be set";
    return 1;
  }
  hexz::InitializeFromConfig(config);

  if (true) {
    RunMultithreadExperiment();
    return 0;
  }
  // Execute
  ABSL_LOG(INFO) << "Worker started with " << config.String();
  if (config.startup_delay_seconds > 0) {
    hexz::internal::RandomDelay(config.startup_delay_seconds);
    ABSL_LOG(INFO) << "Startup delay finished.";
  }
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
  if (config.debug_memory_usage) {
    memmon = std::thread{MemMon};
  }
  hexz::GenerateExamples(config);
  return 0;
}
