#ifndef __HEXZ_WORKER_H__
#define __HEXZ_WORKER_H__

#include "base.h"
#include "grpc_client.h"
#include "mcts.h"

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

class Worker {
 public:
  Worker(const Config& config, TrainingServiceClient& client)
      : config_{config}, client_{client}, execution_id_{RandomUid()} {}

  void Run();

 private:
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

#endif  // __HEXZ_WORKER_H__
