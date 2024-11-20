#include "worker.h"

#include <absl/cleanup/cleanup.h>
#include <absl/log/absl_log.h>
#include <absl/strings/string_view.h>

#include <boost/fiber/all.hpp>
#include <mutex>
#include <random>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

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

bool SameOrNewer(const hexzpb::ModelKey& lhs, const hexzpb::ModelKey& rhs) {
  return lhs.name() == rhs.name() && lhs.checkpoint() <= rhs.checkpoint();
}

// Execution ID used for a "kill" message that will shut down the
// AsyncExampleSender thread.
constexpr absl::string_view kKillMessage = "__KILL_KILL_KILL__";

}  // namespace

AsyncExampleSender::AsyncExampleSender(TrainingServiceClient& client,
                                       Model& model, bool dry_run)
    : dry_run_{dry_run},
      state_{State::PENDING},
      client_{client},
      model_{model} {
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
    const int max_errors = 3;
    int errors = 0;
    while (errors < max_errors) {
      auto req = request_queue_.pop();

      if (req.execution_id() == kKillMessage) {
        ABSL_LOG(INFO) << "AsyncExampleSender: received kill request";
        break;
      }

      if (ProcessRequest(req)) {
        errors = 0;  // Success: reset error counter.
      } else {
        errors++;
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
  bool send_kill_msg = false;
  {
    std::scoped_lock<std::mutex> lk(mut_);
    ABSL_CHECK(state_ != State::PENDING)
        << "AsyncExampleSender::TerminateSenderThread called in state PENDING.";
    if (state_ == State::ACTIVE) {
      // state_ might already be TERMINATED, if sender thread aborted due to RPC
      // errors. So only update if it's still ACTIVE.
      state_ = State::STOPPING;
      send_kill_msg = true;
    }
  }
  if (send_kill_msg) {
    // Send kill message to terminate sender thread.
    hexzpb::AddTrainingExamplesRequest kill_req;
    kill_req.set_execution_id(kKillMessage);
    ABSL_LOG(INFO) << "AsyncExampleSender: terminating sender thread";
    // Cannot call EnqueueRequest here anymore in state STOPPING.
    request_queue_.push(std::move(kill_req));
  }
  if (sender_thread_.joinable()) {
    sender_thread_.join();
  }
}

bool AsyncExampleSender::ProcessRequest(
    const hexzpb::AddTrainingExamplesRequest& req) {
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
    internal::RNG rng;
    float delay = config_.startup_delay_seconds * rng.Uniform();
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
      ABSL_CHECK(fiber_torch_model != nullptr)
          << "suspend_while_training is only supported by the "
             "FiberTorchModel";
    }
    absl::Status status = client_.StreamControlEvents(
        grpc_client_context, request,
        [fiber_torch_model](hexzpb::ControlEvent event) {
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

#ifdef __linux__
  bool pin_threads = config_.pin_threads;
  if (pin_threads) {
    long n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpus >= config_.worker_threads) {
      ABSL_LOG(INFO)
          << "We haz " << n_cpus
          << " online CPUs on this machine. Enabling thread affinity for "
          << config_.worker_threads << " worker threads.";
    } else {
      pin_threads = false;
      ABSL_LOG(WARNING) << "Disabling thread affinity: more threads requested ("
                        << config_.worker_threads << ") than there are CPUs ("
                        << n_cpus << ")";
    }
  }
#endif
  std::vector<std::thread> worker_threads;
  for (int i = 0; i < config_.worker_threads; i++) {
    worker_threads.emplace_back([this, &model, &sender, thread_num = i] {
      // Don't calculate gradients: self-play only needs model evaluation.
      torch::NoGradGuard no_grad;
      // Ensure thread local performance metrics get aggregated into global
      // metrics.
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
#ifdef __linux__
    if (pin_threads) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(i, &cpuset);
      int rc = pthread_setaffinity_np(worker_threads.back().native_handle(),
                                      sizeof(cpu_set_t), &cpuset);
      ABSL_CHECK(rc == 0) << "Failed to set thread affinity for thread " << i;
    }
#endif
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

    const std::string game_id = RandomUid();
    auto examples = mcts.PlayGame(game_id, b, max_runtime_seconds);
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
    req.set_game_id(game_id);
    PopulateWorkerConfig(*req.mutable_worker_config());
    std::move(examples->begin(), examples->end(),
              RepeatedPtrFieldBackInserter(req.mutable_examples()));
    if (!sender.EnqueueRequest(std::move(req))) {
      ABSL_LOG(ERROR) << "Aborting: could not enqueue examples for sending";
      break;
    }

    // Update stats.
    stats_.IncrementExamples(n_examples);
    stats_.IncrementGames(1);
    APMGames().Increment(1);
  }
}

void Worker::PopulateWorkerConfig(hexzpb::WorkerConfig& config) const {
  config.set_device(torch::DeviceTypeName(DeviceType()));
  config.set_worker_threads(config_.worker_threads);
  config.set_fibers_per_thread(config_.fibers_per_thread);
  config.set_prediction_batch_size(config_.prediction_batch_size);
  config.set_runs_per_move(config_.runs_per_move);
  config.set_dirichlet_concentration(config_.dirichlet_concentration);
  config.set_fast_move_prob(config_.fast_move_prob);
  config.set_runs_per_fast_move(config_.runs_per_fast_move);
  // We could also take these values from config_, but since the Node::*
  // values are what actually gets used, let's use those here, too.
  config.set_uct_c(Node::uct_c);
  config.set_initial_root_q_value(Node::initial_root_q_value);
  config.set_initial_q_penalty(Node::initial_q_penalty);
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
    return std::make_unique<FiberTorchModel>(
        model_key, std::move(model), device, config_.prediction_batch_size,
        config_.suspend_while_training);
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
