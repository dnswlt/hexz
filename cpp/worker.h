#pragma once

#include "base.h"
#include "grpc_client.h"
#include "mcts.h"
#include "queue.h"

namespace hexz {

class WorkerStats {
 public:
  // Non-copyable and non-moveable.
  WorkerStats() = default;
  WorkerStats(const WorkerStats&) = delete;
  WorkerStats& operator=(const WorkerStats&) = delete;

  struct Data {
    Data() = default;
    Data(const Data&) = default;
    int examples = 0;
    int games = 0;
    // Unix micros
    int64_t started_micros = 0;
  };

  void SetStartedMicros(int64_t started_micros) {
    std::unique_lock<std::mutex> lk(mut_);
    data_.started_micros = started_micros;
  }
  void IncrementGames(int n) {
    std::unique_lock<std::mutex> lk(mut_);
    data_.games += n;
  }
  void IncrementExamples(int n) {
    std::unique_lock<std::mutex> lk(mut_);
    data_.examples += n;
  }
  Data GetData() {
    std::unique_lock<std::mutex> lk(mut_);
    return data_;
  }

 private:
  std::mutex mut_;
  Data data_;
};


// AsyncExampleSender is used to send examples to the training server
// asynchronously. Its Run() method is supposed to be executed in a separate
// thread. Clients can then pass requests via the EnqueueRequest method from any
// other thread without additional synchronization.
//
// If the training server informs the sender that a newer model is available,
// the sender will update the model.
class AsyncExampleSender {
 public:
  enum class State { PENDING, ACTIVE, STOPPING, TERMINATED };
  // Starts the background sender thread, which will be terminated on
  // destruction of this object.
  AsyncExampleSender(TrainingServiceClient& client, Model& model);
  // Cannot copy instances.
  AsyncExampleSender(const AsyncExampleSender&) = delete;
  AsyncExampleSender& operator=(const AsyncExampleSender&) = delete;
  // Terminates the sender thread.
  ~AsyncExampleSender();

  // Enqueues another request to be sent to the training server.
  bool EnqueueRequest(hexzpb::AddTrainingExamplesRequest&& req);

 private:
  // Start the background thread that will process send requests.
  void StartSenderThread();

  // Put a "kill message" into the queue to shut down the processing thread.
  void TerminateSenderThread();

  bool ProcessRequest(const hexzpb::AddTrainingExamplesRequest& req);

  mutable std::mutex mut_;
  State state_;
  std::thread sender_thread_;
  // Not owned.
  TrainingServiceClient& client_;
  Model& model_;
  ConcurrentQueue<hexzpb::AddTrainingExamplesRequest> request_queue_;
};

class Worker {
 public:
  Worker(const Config& config, TrainingServiceClient& client)
      : config_{config}, client_{client}, execution_id_{RandomUid()} {}

  void Run();

 private:
  void RunSingle(Model& model, AsyncExampleSender& sender);
  torch::DeviceType Device() const;
  std::unique_ptr<Model> CreateModel(hexzpb::ModelKey model_key,
                                     torch::jit::Module&& model);
  std::string execution_id_;
  TrainingServiceClient& client_;
  const Config& config_;
  WorkerStats stats_;
};

void GenerateExamplesMultiThreaded(const Config& config,
                                   TrainingServiceClient& client);

}  // namespace hexz
