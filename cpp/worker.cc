#include "worker.h"

#include <absl/log/absl_log.h>
#include <absl/strings/string_view.h>

#include <mutex>
#include <random>
#include <sstream>

#include "grpc_client.h"
#include "hexz.pb.h"
#include "mcts.h"
#include "queue.h"

namespace hexz {

namespace {
using google::protobuf::RepeatedPtrFieldBackInserter;

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

constexpr absl::string_view kKillMessage = "__KILL_KILL_KILL__";

// AsyncExampleSender is used to send examples to the training server asynchronously.
// Its Run() method is supposed to be executed in a separate thread. 
// Clients can then pass requests via the EnqueueRequest method from any other thread
// without additional synchronization.
//
// If the training server informs the sender that a newer model is available,
// the sender will update the model.
class AsyncExampleSender {
 public:
  explicit AsyncExampleSender(TrainingServiceClient& client, Model& model)
      : client_{client}, model_{model} {}
  // Cannot copy instances.
  AsyncExampleSender(const AsyncExampleSender&) = delete;
  AsyncExampleSender& operator=(const AsyncExampleSender&) = delete;

  void Run() {
    while (true) {
      auto req = request_queue_.pop();
      if (!ProcessRequest(req)) {
        break;
      }
    }
    {
      std::scoped_lock<std::mutex> lk(mut_);
      done_ = true;
    }
    ABSL_LOG(INFO) << "AsyncExampleSender::Run() is done.";
  }

  bool EnqueueRequest(hexzpb::AddTrainingExamplesRequest&& req) {
    {
      std::scoped_lock<std::mutex> lk(mut_);
      if (done_) {
        return false;
      }
    }
    request_queue_.push(std::move(req));
    return true;
  }

  // Put a "kill message" into the queue to  shut down the  processing thread.
  void Terminate() {
    hexzpb::AddTrainingExamplesRequest kill_req;
    kill_req.set_execution_id(kKillMessage);
    EnqueueRequest(std::move(kill_req));
  }

 private:
  bool ProcessRequest(const hexzpb::AddTrainingExamplesRequest& req) {
    if (req.execution_id() == kKillMessage) {
        ABSL_LOG(ERROR) << "AsyncExampleSender: received kill request";
        return false;
    }
    auto resp = client_.AddTrainingExamples(req);
    if (!resp.ok()) {
      ABSL_LOG(ERROR) << "Failed to send examples: " << resp.status();
      return false;
    }
    switch (resp->status()) {
      case hexzpb::AddTrainingExamplesResponse::ACCEPTED:
        // Happy path.
        ABSL_LOG(INFO) << "Successfully sent " << req.examples_size()
                       << " examples to training server at "
                       << client_.RemoteAddr();
        break;
      case hexzpb::AddTrainingExamplesResponse::REJECTED_AT_CAPACITY:
        // For now, immediately exit if server is at capacity.
        // There are probably too many worker (threads) sending examples.
        ABSL_LOG(ERROR) << "Server is at capacity: " << resp->error_message();
        return false;
      case hexzpb::AddTrainingExamplesResponse::REJECTED_WRONG_MODEL:
        // Since the server should accept both the previous and the latest
        // model version, this should be a rare event, and we should
        // probably tune our batch size. Individual workers are not able to
        // generate new examples before the model got updated twice.
        {
          const auto& model_id =
              ModelId(req.examples(req.examples_size() - 1).model_key());
          ABSL_LOG(ERROR)
              << "Server rejected out examples due to an old model. Sent: "
              << model_id << ", want: " << ModelId(resp->latest_model());
          return false;
        }
      default:
        // Unknown status code => exit.
        ABSL_LOG(ERROR) << "Received unexpected response status: "
                        << hexzpb::AddTrainingExamplesResponse::Status_Name(
                               resp->status())
                        << ": " << resp->error_message();
        return false;
    }

    // Update model if necessary.
    if (!SameOrNewer(resp->latest_model(), model_.Key())) {
      auto old_key = model_.Key();
      auto km = client_.FetchLatestModel(resp->latest_model().name());
      if (!km.ok()) {
        ABSL_LOG(ERROR) << "Failed to fetch latest model: " << km.status();
        return false;
      }
      const auto& [latest_key, latest_model] = *km;
      model_.UpdateModel(latest_key, latest_model);
      ABSL_LOG(INFO) << "Updated model from " << ModelId(old_key) << " to "
                     << ModelId(latest_key);
    }
    return true;
  }

  // Not owned.
  TrainingServiceClient& client_;
  Model& model_;
  mutable std::mutex mut_;
  bool done_ = false;
  ConcurrentQueue<hexzpb::AddTrainingExamplesRequest> request_queue_;
};

void GenerateExamplesMultiThreaded(const Config& config,
                                   TrainingServiceClient& client) {
  WorkerStats stats;
  const int64_t started_micros = UnixMicros();
  const int64_t end_micros =
      started_micros +
      static_cast<int64_t>(config.max_runtime_seconds) * 1'000'000;
  const std::string execution_id = RandomUid();
  ABSL_LOG(INFO) << "Generating examples using execution_id " << execution_id;

  // Leave model_name empty to fetch whatever the training server has available.
  auto km = client.FetchLatestModel(/*model_name=*/"");
  if (!km.ok()) {
    ABSL_LOG(ERROR) << "Failed to fetch latest model: " << km.status();
    return;
  }
  const auto& [initial_model_key, initial_model] = *km;
  torch::DeviceType device = torch::kCPU;
  if (config.device == "mps") {
    device = torch::kMPS;
  } else if (config.device == "cuda") {
    device = torch::kCUDA;
  }

  std::unique_ptr<Model> model;
  if (config.worker_threads > 1 && device != torch::kCPU) {
    // Using the batched model is only useful if multiple threads use the GPU.
    constexpr int64_t timeout_micros = 1'000'000;
    ABSL_LOG(INFO) << "Using BatchedTorchModel for " << config.worker_threads
                   << " threads on device " << config.device;
    model = std::make_unique<BatchedTorchModel>(
        initial_model_key, initial_model, device, config.worker_threads,
        timeout_micros);
  } else {
    ABSL_LOG(INFO) << "Using TorchModel for " << config.worker_threads
                   << " threads on device " << config.device;
    model = std::make_unique<TorchModel>(initial_model_key, initial_model);
  }
  // Used to send examples and update the model asynchronously.
  AsyncExampleSender sender(client, *model);
  std::thread sender_thread(&AsyncExampleSender::Run, &sender);
  sender.Terminate();

  stats.Start();
  std::vector<std::thread> worker_threads;
  for (int i = 0; i < config.worker_threads; i++) {
    worker_threads.emplace_back([&, thread_num = i] {
      Perfm::ThreadScope perfm;
      auto guard = model->RegisterThread();
      // Delay startup if requested.
      if (config.startup_delay_seconds > 0) {
        float delay = config.startup_delay_seconds * internal::UnitRandom();
        ABSL_LOG(INFO) << "Delaying startup by " << delay << " seconds.";
        std::this_thread::sleep_for(std::chrono::duration<float>(delay));
      }

      int max_games = config.max_games > 0 ? config.max_games
                                           : std::numeric_limits<int>::max();
      for (int i = 0; i < max_games; i++) {
        int64_t now = UnixMicros();
        if (now >= end_micros) {
          break;  // Time's up
        }
        NeuralMCTS mcts{*model, std::make_unique<RandomPlayoutRunner>(),
                        config};
        Board b = Board::RandomBoard();
        int64_t max_runtime_seconds =
            config.max_runtime_seconds - (now - started_micros) / 1'000'000;

        auto examples = mcts.PlayGame(b, max_runtime_seconds);
        if (!examples.ok()) {
          if (!absl::IsDeadlineExceeded(examples.status())) {
            ABSL_LOG(ERROR) << "Aborting: PlayGame returned an error: "
                            << examples.status();
          }
          break;
        }
        const int n_examples = examples->size();

        ABSL_CHECK(n_examples > 0)
            << "Played a game that yielded no examples?!";
        const std::string model_id = ModelId(examples->back().model_key());

        hexzpb::AddTrainingExamplesRequest req;
        req.set_execution_id(execution_id);
        std::move(examples->begin(), examples->end(),
                  RepeatedPtrFieldBackInserter(req.mutable_examples()));
        if (!sender.EnqueueRequest(std::move(req))) {
          ABSL_LOG(ERROR) << "Aborting: could not enqueue examples for sending";
          break;
        }

        // Update stats.
        stats.IncrementExamples(n_examples);
        stats.IncrementGames(1);
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
  sender.Terminate();
  if (sender_thread.joinable()) {
    sender_thread.join();
  }
  // Print stats.
  auto stats_data = stats.data();
  auto d = static_cast<double>(UnixMicros() - stats_data.started) / 1e6;
  ABSL_LOG(INFO) << "Generated " << stats_data.games << " games and "
                 << stats_data.examples << " examples in " << d << " seconds ("
                 << (stats_data.examples / d) << " examples/s, "
                 << (stats_data.games / d) << " games/s)";
}

}  // namespace hexz