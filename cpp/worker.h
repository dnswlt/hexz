#ifndef __HEXZ_WORKER_H__
#define __HEXZ_WORKER_H__

#include "base.h"
#include "grpc_client.h"

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
    int64_t started = 0;
  };

  void Start() {
    std::unique_lock<std::mutex> lk(mut_);
    data_.started = UnixMicros();
  }
  void IncrementGames(int n) {
    std::unique_lock<std::mutex> lk(mut_);
    data_.games += n;
  }
  void IncrementExamples(int n) {
    std::unique_lock<std::mutex> lk(mut_);
    data_.examples += n;
  }
  Data data() {
    std::unique_lock<std::mutex> lk(mut_);
    return data_;
  }

 private:
  std::mutex mut_;
  Data data_;
};

void GenerateExamplesMultiThreaded(const Config& config, TrainingServiceClient& client);

}  // namespace hexz

#endif  // __HEXZ_WORKER_H__
