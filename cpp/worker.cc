#include "worker.h"

#include <absl/cleanup/cleanup.h>
#include <absl/log/absl_log.h>
#include <absl/strings/string_view.h>

#include <boost/fiber/all.hpp>
#include <mutex>
#include <random>

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

constexpr absl::string_view kKillMessage = "__KILL_KILL_KILL__";

}  // namespace

AsyncExampleSender::AsyncExampleSender(TrainingServiceClient& client,
                                       Model& model, bool dry_run)
    : client_{client},
      model_{model},
      state_{State::PENDING},
      dry_run_{dry_run} {
  StartSenderThread();
}
AsyncExampleSender::~AsyncExampleSender() { TerminateSenderThread(); }

// Enqueues another request to be sent to the training server.
bool AsyncExampleSender::EnqueueRequest(
    hexzpb::AddTrainingExamplesRequest&& req) {
  {
    std::scoped_lock<std::mutex> lk(mut_);
    if (state_ != State::ACTIVE) {
      return false;
    }
  }
  request_queue_.push(std::move(req));
  return true;
}

// Start the background thread that will process send requests.
void AsyncExampleSender::StartSenderThread() {
  std::scoped_lock<std::mutex> lk(mut_);
  ABSL_CHECK(state_ == State::PENDING)
      << "AsyncExampleSender::Start must only be called once.";
  sender_thread_ = std::thread([this] {
    while (true) {
      auto req = request_queue_.pop();
      if (!ProcessRequest(req)) {
        break;
      }
    }
    {
      std::scoped_lock<std::mutex> lk(mut_);
      state_ = State::TERMINATED;
    }
    ABSL_LOG(INFO) << "AsyncExampleSender sender thread is done.";
  });
  state_ = State::ACTIVE;
}

// Put a "kill message" into the queue to shut down the processing thread.
void AsyncExampleSender::TerminateSenderThread() {
  {
    std::scoped_lock<std::mutex> lk(mut_);
    if (state_ != State::ACTIVE) {
      return;
    }
    state_ = State::STOPPING;
  }
  ABSL_LOG(INFO) << "AsyncExampleSender: terminating sender thread";
  // Send kill message to terminate sender thread.
  hexzpb::AddTrainingExamplesRequest kill_req;
  kill_req.set_execution_id(kKillMessage);
  // Cannot call EnqueueRequest here anymore in state STOPPING.
  request_queue_.push(std::move(kill_req));

  if (sender_thread_.joinable()) {
    sender_thread_.join();
  }
}

bool AsyncExampleSender::ProcessRequest(
    const hexzpb::AddTrainingExamplesRequest& req) {
  if (req.execution_id() == kKillMessage) {
    ABSL_LOG(ERROR) << "AsyncExampleSender: received kill request";
    return false;
  }
  if (dry_run_) {
    ABSL_LOG(INFO) << "AsyncExampleSender is in dry_run mode. Ignoring "
                   << req.examples_size() << " examples";
    return true;
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
    const auto old_key = model_.Key();
    auto km = client_.FetchLatestModel(resp->latest_model().name());
    if (!km.ok()) {
      ABSL_LOG(ERROR) << "Failed to fetch latest model: " << km.status();
      return false;
    }
    auto& [latest_key, latest_model] = *km;
    model_.UpdateModel(latest_key, std::move(latest_model));
    ABSL_LOG(INFO) << "Updated model from " << ModelId(old_key) << " to "
                   << ModelId(latest_key);
  }
  return true;
}

void Worker::Run() {
  ABSL_LOG(INFO) << "Generating examples using execution_id " << execution_id_;

  // Delay startup if requested.
  if (config_.startup_delay_seconds > 0) {
    float delay = config_.startup_delay_seconds * internal::UnitRandom();
    ABSL_LOG(INFO) << "Delaying startup by " << delay << " seconds.";
    std::this_thread::sleep_for(std::chrono::duration<float>(delay));
  }

  // Leave model_name empty to fetch whatever the training server has
  // available.
  auto km = client_.FetchLatestModel(/*model_name=*/"");
  if (!km.ok()) {
    ABSL_LOG(ERROR) << "Failed to fetch latest model: " << km.status();
    return;
  }
  auto& [initial_model_key, initial_model] = *km;
  ABSL_LOG(INFO) << "Fetched latest model: " << initial_model_key.name() << ":"
                 << initial_model_key.checkpoint();

  std::unique_ptr<Model> model =
      CreateModel(initial_model_key, std::move(initial_model));

  // Subscribe to server's ControlEvents.
  grpc::ClientContext grpc_client_context;
  std::thread control_events_thread([this, &grpc_client_context, &model] {
    hexzpb::ControlRequest request;
    request.set_execution_id(execution_id_);
    FiberTorchModel* fiber_torch_model = nullptr;
    if (config_.suspend_while_training) {
      fiber_torch_model = dynamic_cast<FiberTorchModel*>(model.get());
    }
    absl::Status status = client_.StreamControlEvents(
        grpc_client_context, request,
        [&model, fiber_torch_model](hexzpb::ControlEvent event) {
          if (!fiber_torch_model) {
            // Suspension is only supported by FiberTorchModel.
            return;
          }
          if (event.has_training_started()) {
            fiber_torch_model->Suspend();
          } else if (event.has_training_done()) {
            fiber_torch_model->Resume();
          }
        });
    ABSL_LOG(INFO) << "ControlEvent streaming RPC finished with status "
                   << status;
  });
  absl::Cleanup cleanup_grpc = [&control_events_thread, &grpc_client_context] {
    ABSL_LOG(INFO) << "Cancelling ControlEvent streaming RPC";
    grpc_client_context.TryCancel();
    if (control_events_thread.joinable()) {
      control_events_thread.join();
    }
  };

  // Used to send examples and update the model asynchronously.
  AsyncExampleSender sender(client_, *model, config_.dry_run);

  stats_.SetStartedMicros(UnixMicros());

  std::vector<std::thread> worker_threads;
  for (int i = 0; i < config_.worker_threads; i++) {
    worker_threads.emplace_back([this, &model, &sender, thread_num = i] {
      Perfm::ThreadScope perfm;
      if (config_.fibers_per_thread > 0) {
        // Run with fibers.
        std::vector<boost::fibers::fiber> fibers;
        fibers.reserve(config_.fibers_per_thread);
        for (int j = 0; j < config_.fibers_per_thread; ++j) {
          fibers.emplace_back([&, fiber_num = j] {
            auto token = model->RegisterThread();
            RunSingle(*model, sender);
            ABSL_LOG(INFO) << "Fiber " << thread_num << ":" << fiber_num
                           << " is done";
          });
        }
        for (auto& fiber : fibers) {
          if (fiber.joinable()) {
            fiber.join();
          }
        }
      } else {
        // No fibers, run directly in thread.
        auto guard = model->RegisterThread();
        RunSingle(*model, sender);
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
  auto stats_data = stats_.GetData();
  auto d = static_cast<double>(UnixMicros() - stats_data.started_micros) / 1e6;
  ABSL_LOG(INFO) << "Generated " << stats_data.games << " games and "
                 << stats_data.examples << " examples in " << d << " seconds ("
                 << (stats_data.examples / d) << " examples/s, "
                 << (stats_data.games / d) << " games/s)";
}

void Worker::RunSingle(Model& model, AsyncExampleSender& sender) {
  const int64_t started_micros = UnixMicros();
  // Need to cast here to avoid int overflow.
  const int64_t end_micros =
      started_micros +
      static_cast<int64_t>(config_.max_runtime_seconds) * 1'000'000;

  int max_games = config_.max_games > 0 ? config_.max_games
                                        : std::numeric_limits<int>::max();
  for (int i = 0; i < max_games; i++) {
    int64_t now = UnixMicros();
    if (now >= end_micros) {
      break;  // Time's up
    }
    NeuralMCTS mcts{model, std::make_unique<RandomPlayoutRunner>(), config_};
    Board b = Board::RandomBoard();
    int64_t max_runtime_seconds =
        config_.max_runtime_seconds - (now - started_micros) / 1'000'000;

    auto examples = mcts.PlayGame(b, max_runtime_seconds);
    if (!examples.ok()) {
      if (!absl::IsDeadlineExceeded(examples.status())) {
        ABSL_LOG(ERROR) << "Aborting: PlayGame returned an error: "
                        << examples.status();
      }
      break;
    }
    const int n_examples = examples->size();

    ABSL_CHECK(n_examples > 0) << "Played a game that yielded no examples?!";
    const std::string model_id = ModelId(examples->back().model_key());

    hexzpb::AddTrainingExamplesRequest req;
    req.set_execution_id(execution_id_);
    std::move(examples->begin(), examples->end(),
              RepeatedPtrFieldBackInserter(req.mutable_examples()));
    if (!sender.EnqueueRequest(std::move(req))) {
      ABSL_LOG(ERROR) << "Aborting: could not enqueue examples for sending";
      break;
    }

    // Update stats.
    stats_.IncrementExamples(n_examples);
    stats_.IncrementGames(1);
  }
}

torch::DeviceType Worker::DeviceType() const {
  if (config_.device == "mps") {
    return torch::kMPS;
  } else if (config_.device == "cuda") {
    return torch::kCUDA;
  }
  return torch::kCPU;
}

std::unique_ptr<Model> Worker::CreateModel(hexzpb::ModelKey model_key,
                                           torch::jit::Module&& model) {
  torch::DeviceType device = DeviceType();
  if (config_.fibers_per_thread > 0) {
    // Use the fiber-based model.
    ABSL_LOG(INFO) << "Using FiberTorchModel for " << config_.worker_threads
                   << " threads and " << config_.fibers_per_thread
                   << " fibers per thread on device " << config_.device
                   << " with batch size " << config_.prediction_batch_size;
    // TODO: Only enable suspension when running on same machine as training
    // server.
    return std::make_unique<FiberTorchModel>(
        model_key, std::move(model), device, config_.prediction_batch_size,
        true);
  } else if (config_.worker_threads > 1 && device != torch::kCPU) {
    // Using the batched model is only useful if multiple threads use the GPU.
    ABSL_CHECK(config_.prediction_batch_size <= config_.worker_threads)
        << "BatchedTorchModel would deadlock";
    constexpr int64_t timeout_micros = 1'000'000;
    ABSL_LOG(INFO) << "Using BatchedTorchModel for " << config_.worker_threads
                   << " threads on device " << config_.device;
    return std::make_unique<BatchedTorchModel>(
        model_key, std::move(model), device, config_.prediction_batch_size,
        timeout_micros);
  }
  // Use the ole' unbatched, one prediction at a time model.
  ABSL_LOG(INFO) << "Using TorchModel for " << config_.worker_threads
                 << " threads on device " << config_.device;
  return std::make_unique<TorchModel>(model_key, std::move(model), device);
}

}  // namespace hexz
