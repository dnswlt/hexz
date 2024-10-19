#include "worker.h"

#include <absl/log/absl_log.h>

#include <mutex>
#include <random>
#include <sstream>

#include "grpc_client.h"
#include "hexz.pb.h"
#include "mcts.h"

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

void GenerateExamplesMultiThreaded(const Config& config, TrainingServiceClient& client) {
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

  // Never let a thread wait for its prediction to run for more than a 1s.
  // Mutex that protects updates to the model
  std::mutex model_update_mut;

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

        hexzpb::AddTrainingExamplesRequest req;
        req.set_execution_id(execution_id);
        std::move(examples->begin(), examples->end(),
                  RepeatedPtrFieldBackInserter(req.mutable_examples()));
        auto resp = client.AddTrainingExamples(req);

        if (!resp.ok()) {
          ABSL_LOG(ERROR) << "Failed to send examples: " << resp.status();
          return;
        }
        switch (resp->status()) {
          case hexzpb::AddTrainingExamplesResponse::ACCEPTED:
            // Happy path.
            ABSL_LOG(INFO) << "Successfully sent " << n_examples
                           << " examples to training server at "
                           << config.training_server_addr;
            break;
          case hexzpb::AddTrainingExamplesResponse::REJECTED_AT_CAPACITY:
            // For now, immediately exit if server is at capacity.
            // There are probably too many worker (threads) sending examples.
            ABSL_LOG(ERROR)
                << "Server is at capacity: " << resp->error_message();
            return;
          case hexzpb::AddTrainingExamplesResponse::REJECTED_WRONG_MODEL:
            // Since the server should accept both the previous and the latest
            // model version, this should be a rare event, and we should
            // probably tune our batch size. Individual workers are not able to
            // generate new examples before the model got updated twice.
            ABSL_LOG(ERROR)
                << "Server rejected out examples due to an old model. Sent: "
                << model_id << ", want: " << ModelId(resp->latest_model());
            return;
          default:
            // Unknown status code => exit.
            ABSL_LOG(ERROR) << "Received unexpected response status: "
                            << hexzpb::AddTrainingExamplesResponse::Status_Name(
                                   resp->status())
                            << ": " << resp->error_message();
            return;
        }
        // Update stats.
        stats.IncrementExamples(n_examples);
        stats.IncrementGames(1);
        // Check if we need to update to a new model.
        {
          std::scoped_lock<std::mutex> lk(model_update_mut);
          if (!SameOrNewer(resp->latest_model(), model->Key())) {
            auto old_key = model->Key();
            auto km = client.FetchLatestModel(resp->latest_model().name());
            if (!km.ok()) {
              ABSL_LOG(ERROR)
                  << "Failed to fetch latest model: " << km.status();
              return;
            }
            const auto& [latest_key, latest_model] = *km;
            model->UpdateModel(latest_key, latest_model);
            ABSL_LOG(INFO) << "Updated model from " << ModelId(old_key)
                           << " to " << ModelId(latest_key);
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
  // Print stats.
  auto stats_data = stats.data();
  auto d = static_cast<double>(UnixMicros() - stats_data.started) / 1e6;
  ABSL_LOG(INFO) << "Generated " << stats_data.games << " games and "
                 << stats_data.examples << " examples in " << d << " seconds ("
                 << (stats_data.examples / d) << " examples/s, "
                 << (stats_data.games / d) << " games/s)";
}

}  // namespace hexz