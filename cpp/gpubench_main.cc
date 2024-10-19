#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/log/absl_log.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>

#include <boost/fiber/all.hpp>
#include <iostream>
#include <string>

#include "base.h"
#include "batch.h"
#include "grpc_client.h"
#include "hexz.pb.h"
#include "mcts.h"

ABSL_FLAG(std::string, mode, "single",
          "GPU benchmark mode (single, batch, task, fiber)");
ABSL_FLAG(std::string, device, "", "PyTorch device (cpu, cuda, mps)");
ABSL_FLAG(int, rounds, 0, "number of rounds to run in a GPU benchmark");
ABSL_FLAG(std::string, model_path, "",
          "path to a torch::jit::load'able model (for benchmarks)");
ABSL_FLAG(int, worker_threads, 0, "number of threads");
ABSL_FLAG(int, prediction_batch_size, 0,
          "batch size for GPU model predictions");

namespace hexz {

struct Options {
  std::string device = "cpu";
  int rounds = 1;
  std::string model_path;
  int worker_threads = 1;
  int prediction_batch_size = 1;

  static Options FromFlags() {
    Options opts;
    if (std::string device = absl::GetFlag(FLAGS_device); device != "") {
      opts.device = device;
    }
    if (int rounds = absl::GetFlag(FLAGS_rounds); rounds > 0) {
      opts.rounds = rounds;
    }
    if (std::string model_path = absl::GetFlag(FLAGS_model_path);
        model_path != "") {
      opts.model_path = model_path;
    }
    if (int worker_threads = absl::GetFlag(FLAGS_worker_threads);
        worker_threads > 0) {
      opts.worker_threads = worker_threads;
    }
    if (int prediction_batch_size = absl::GetFlag(FLAGS_prediction_batch_size);
        prediction_batch_size > 0) {
      opts.prediction_batch_size = prediction_batch_size;
    }
    return opts;
  }
};

template <typename T>
class BoundedConcurrentQueue {
 public:
  // Constructor: Initializes the queue with a given capacity
  explicit BoundedConcurrentQueue(size_t capacity) : max_capacity(capacity) {}

  // Pushes an item to the queue. Blocks if the queue is full.
  void push(const T& item) {
    std::unique_lock<std::mutex> lock(mtx);
    cv_not_full.wait(lock, [this] {
      return q.size() < max_capacity;
    });  // Wait if the queue is full
    q.push(item);
    cv_not_empty.notify_one();  // Notify consumers that an item is available
  }

  // Pushes an item to the queue using move semantics. Blocks if the queue is
  // full.
  void push(T&& item) {
    std::unique_lock<std::mutex> lock(mtx);
    cv_not_full.wait(lock, [this] {
      return q.size() < max_capacity;
    });  // Wait if the queue is full
    q.push(std::move(item));
    cv_not_empty.notify_one();  // Notify consumers that an item is available
  }

  // Pops an item from the queue. Blocks if the queue is empty.
  T pop() {
    std::unique_lock<std::mutex> lock(mtx);
    cv_not_empty.wait(
        lock, [this] { return !q.empty(); });  // Wait if the queue is empty
    T item = std::move(q.front());
    q.pop();
    cv_not_full.notify_one();  // Notify producers that space is available
    return item;
  }

  // Tries to pop an item without blocking. Returns false if the queue is empty.
  bool try_pop(T& item) {
    std::lock_guard<std::mutex> lock(mtx);
    if (q.empty()) {
      return false;
    }
    item = std::move(q.front());
    q.pop();
    cv_not_full.notify_one();  // Notify producers that space is available
    return true;
  }

  // Checks if the queue is empty
  bool empty() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.empty();
  }

  // Checks if the queue is full
  bool full() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.size() >= max_capacity;
  }

  // Returns the size of the queue
  size_t size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.size();
  }

 private:
  std::queue<T> q;
  size_t max_capacity;     // Maximum number of elements the queue can hold
  mutable std::mutex mtx;  // Mutex for synchronizing access
  std::condition_variable
      cv_not_full;  // Condition variable for "not full" state
  std::condition_variable
      cv_not_empty;  // Condition variable for "not empty" state
};

template <typename T>
class ConcurrentQueue {
 public:
  // Adds an element to the queue
  void push(const T& item) {
    std::lock_guard<std::mutex> lock(mtx);
    q.push(item);
    cv.notify_one();  // Notify one waiting thread
  }

  // Adds an element using move semantics
  void push(T&& item) {
    std::lock_guard<std::mutex> lock(mtx);
    q.push(std::move(item));
    cv.notify_one();  // Notify one waiting thread
  }

  // Pops an element from the queue
  // Blocks if the queue is empty
  T pop() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return !q.empty(); });  // Wait until there's data
    T item = std::move(q.front());
    q.pop();
    return item;
  }

  // Moves up to n items from the queue into the given vector dst.
  // Blocks if the queue is empty.
  void pop_n(std::vector<T>& dst, int n) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return !q.empty(); });  // Wait until there's data
    while (n > 0 && !q.empty()) {
      dst.push_back(std::move(q.front()));
      q.pop();
      n--;
    }
  }

  // Tries to pop an element without blocking
  bool try_pop(T& item) {
    std::lock_guard<std::mutex> lock(mtx);
    if (q.empty()) {
      return false;
    }
    item = std::move(q.front());
    q.pop();
    return true;
  }

  // Check if the queue is empty
  bool empty() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.empty();
  }

  // Get the size of the queue
  size_t size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.size();
  }

 private:
  std::queue<T> q;
  mutable std::mutex mtx;      // Mutex for synchronizing access
  std::condition_variable cv;  // Condition variable to signal waiting threads
};

///////////////////////////////////////////////////////////////////////////////
// TASK
///////////////////////////////////////////////////////////////////////////////

struct GPUPipelineInput {
  torch::Tensor board;
  torch::Tensor action_mask;
  bool done = false;
};

void RunGPUBenchmarkMultiTasked(Options options) {
  EmbeddedTrainingServiceClient client(options.model_path);
  auto km = client.FetchLatestModel("");
  if (!km.ok()) {
    ABSL_LOG(ERROR) << "Failed to fetch model: " << km.status();
    return;
  }
  auto& [key, model] = *km;
  auto device = torch::kCPU;
  if (options.device == "mps") {
    device = torch::kMPS;
  } else if (options.device == "cuda") {
    device = torch::kCUDA;
  }
  model.to(device);

  const int n_rounds = options.rounds;

  BoundedConcurrentQueue<GPUPipelineInput> input_queue(
      options.prediction_batch_size * 2);
  ConcurrentQueue<Model::Prediction> result_queue;

  std::thread gpu_pipeline_thread([&] {
    std::vector<torch::Tensor> boards;
    boards.reserve(options.prediction_batch_size);
    std::vector<torch::Tensor> action_masks;
    action_masks.reserve(options.prediction_batch_size);
    torch::NoGradGuard no_grad;

    for (;;) {
      GPUPipelineInput input = input_queue.pop();
      if (input.done) {
        return;
      }
      boards.emplace_back(std::move(input.board));
      action_masks.emplace_back(std::move(input.action_mask));
      if (boards.size() == options.prediction_batch_size) {
        std::vector<torch::jit::IValue> model_inputs = {
            torch::stack(boards).to(device),
            torch::stack(action_masks).to(device),
        };
        auto pred = model.forward(model_inputs);
        const auto output_tuple = pred.toTuple();
        // Read logits back to CPU / main memory.
        const auto probs =
            output_tuple->elements()[0].toTensor().exp().to(torch::kCPU);
        const auto values =
            output_tuple->elements()[1].toTensor().to(torch::kCPU);
        for (int i = 0; i < boards.size(); i++) {
          result_queue.push(Model::Prediction{
              .move_probs = probs.index({i}).reshape({2, 11, 10}),
              .value = values.index({i}).item<float>(),
          });
        }
        boards.clear();
        action_masks.clear();
      }
    }
  });

  std::vector<std::thread> worker_threads;
  std::mutex mut;
  int op_count = 0;
  const int n_warmups =
      std::max(1024, options.rounds * options.prediction_batch_size / 10);
  const int n_runs = n_warmups + n_rounds * options.prediction_batch_size;
  float total_sum = 0;
  int64_t t_start = 0;
  for (int i = 0; i < options.worker_threads; i++) {
    worker_threads.emplace_back([&, thread_num = i] {
      for (;;) {
        Model::Prediction pred;
        float sum = 0;
        while (result_queue.try_pop(pred)) {
          std::scoped_lock<std::mutex> lk(mut);
          op_count++;
          if (op_count >= n_runs) {
            // Generated enough predictions, time so say goodbye.
            total_sum += sum;
            return;
          } else if (t_start == 0 && op_count >= n_warmups) {
            t_start = UnixMicros();
          }
          sum += pred.value;
        }
        torch::Tensor tb =
            torch::randn({11, 11, 10}, torch::dtype(torch::kFloat32));
        torch::Tensor ta = (torch::rand({2, 11, 10}) < 0.5);
        input_queue.push(GPUPipelineInput{
            .board = tb,
            .action_mask = ta,
        });
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
  float final_sum;
  {
    std::scoped_lock<std::mutex> l(mut);
    final_op_count = op_count - n_warmups;
    final_sum = total_sum;
  }
  // Only print the final_sum to avoid dead code elimination.
  ABSL_LOG(INFO) << "Workers done. Value sum=" << final_sum;
  // Hack: Add one more item to wake up and terminate the gpu_pipeline_thread.
  input_queue.push(GPUPipelineInput{.done = true});
  if (gpu_pipeline_thread.joinable()) {
    gpu_pipeline_thread.join();
  }
  float ops_per_sec =
      static_cast<float>(final_op_count) / (t_end - t_start) * 1e6;
  ABSL_LOG(INFO) << "Performed " << final_op_count << " computations in "
                 << (t_end - t_start) / 1000 << "ms (" << ops_per_sec
                 << " ops/s). batch size:" << options.prediction_batch_size
                 << ", threads:" << options.worker_threads;
}

///////////////////////////////////////////////////////////////////////////////
// BATCH
///////////////////////////////////////////////////////////////////////////////

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

void RunGPUBenchmarkBatch(Options options) {
  ABSL_CHECK(options.worker_threads >= options.prediction_batch_size);
  EmbeddedTrainingServiceClient client(options.model_path);
  auto km = client.FetchLatestModel("");
  if (!km.ok()) {
    ABSL_LOG(ERROR) << "Failed to fetch model: " << km.status();
    return;
  }
  auto& [key, model] = *km;
  auto device = torch::kCPU;
  if (options.device == "mps") {
    device = torch::kMPS;
  } else if (options.device == "cuda") {
    device = torch::kCUDA;
  }
  GPUBenchmarkFn fn(model, device, options.prediction_batch_size);
  const int n_rounds = options.rounds;
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
  for (int i = 0; i < options.worker_threads; i++) {
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
                 << " ops/s). batch size:" << options.prediction_batch_size
                 << ", threads:" << options.worker_threads;
}

///////////////////////////////////////////////////////////////////////////////
// SINGLE
///////////////////////////////////////////////////////////////////////////////

void RunGPUBenchmark(Options options) {
  EmbeddedTrainingServiceClient client(options.model_path);
  auto km = client.FetchLatestModel("");
  if (!km.ok()) {
    ABSL_LOG(ERROR) << "Failed to fetch model: " << km.status();
    return;
  }
  auto& [key, model] = *km;
  auto device = torch::kCPU;
  if (options.device == "mps") {
    device = torch::kMPS;
  } else if (options.device == "cuda") {
    device = torch::kCUDA;
  }
  model.to(device);
  torch::NoGradGuard no_grad;
  std::vector<int> batch_sizes;
  for (int b = 1; b < options.prediction_batch_size; b *= 2) {
    batch_sizes.push_back(b);
  }
  batch_sizes.push_back(options.prediction_batch_size);
  const int n_rounds = options.rounds;
  for (int batch_size : batch_sizes) {
    int64_t t_start = 0;
    const int n_warmups = std::max(10, n_rounds / 10);
    const int n_runs = n_warmups + n_rounds;
    float sum = 0;
    torch::Tensor tb =
        torch::randn({batch_size, 11, 11, 10}, torch::dtype(torch::kFloat32));
    torch::Tensor ta = (torch::rand({batch_size, 2, 11, 10}) < 0.5);
    for (int i = 0; i < n_runs; i++) {
      if (i == n_warmups) {
        // Warmup done, start measuring time.
        t_start = UnixMicros();
      }
      std::vector<torch::jit::IValue> inputs{tb.to(device), ta.to(device)};
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
    ABSL_LOG(INFO) << "batch_size=" << batch_size << ": finished " << n_rounds
                   << " iterations in " << (duration / 1000) << "ms ("
                   << (static_cast<float>(n_rounds * batch_size) / duration *
                       1e6)
                   << " ops/s)";
  }
}

///////////////////////////////////////////////////////////////////////////////
// FIBER
///////////////////////////////////////////////////////////////////////////////

// Shared data structure for request queue
struct FiberRequest {
  // ID of the fiber making the request (only used for debugging).
  int fiber_id;
  torch::Tensor board;
  torch::Tensor action_mask;
  // Promise to send the result back to the fiber.
  boost::fibers::promise<Model::Prediction> result_promise;
  // The final request sets this to true to terminate the GPU pipeline thread.
  bool done = false;
};

void GPUPipelineThread(torch::jit::Module& model,
                       ConcurrentQueue<FiberRequest>& request_queue,
                       Options options) {
  const int batch_size = options.prediction_batch_size;
  while (true) {
    std::vector<FiberRequest> batch;

    // Collect requests for batching
    while (batch.size() < batch_size) {
      request_queue.pop_n(batch, batch_size - batch.size());
      if (batch.back().done) {
        ABSL_LOG(INFO) << "GPU pipeline thread received final request "
                          "(done=true). Exiting.\n";
        return;
      }
    }

    std::vector<torch::Tensor> boards;
    boards.reserve(options.prediction_batch_size);
    std::vector<torch::Tensor> action_masks;
    action_masks.reserve(options.prediction_batch_size);
    torch::NoGradGuard no_grad;

    for (auto& req : batch) {
      boards.emplace_back(std::move(req.board));
      action_masks.emplace_back(std::move(req.action_mask));
    }

    std::vector<torch::jit::IValue> model_inputs = {
        torch::stack(boards).to(options.device),
        torch::stack(action_masks).to(options.device),
    };
    auto pred = model.forward(model_inputs);

    const auto output_tuple = pred.toTuple();
    // Read logits back to CPU / main memory.
    const auto probs =
        output_tuple->elements()[0].toTensor().exp().to(torch::kCPU);
    const auto values = output_tuple->elements()[1].toTensor().to(torch::kCPU);

    for (int i = 0; i < batch.size(); i++) {
      batch[i].result_promise.set_value(Model::Prediction{
          .move_probs = probs.index({i}).reshape({2, 11, 10}),
          .value = values.index({i}).item<float>(),
      });
    }
  }
}

void FiberWorker(int fiber_id, ConcurrentQueue<FiberRequest>& request_queue,
                 int n_rounds) {
  torch::Tensor tb = torch::randn({11, 11, 10}, torch::dtype(torch::kFloat32));
  torch::Tensor ta = (torch::rand({2, 11, 10}) < 0.5);

  float sum = 0;
  for (int i = 0; i < n_rounds; i++) {
    boost::fibers::promise<Model::Prediction> promise;
    auto result = promise.get_future();

    request_queue.push(FiberRequest{
        .fiber_id = fiber_id,
        .board = tb,
        .action_mask = ta,
        .result_promise = std::move(promise),
    });

    Model::Prediction pred = result.get();
    sum += pred.value;
  }
}

void RunGPUBenchmarkFiber(Options options) {
  ABSL_CHECK(options.worker_threads <= options.prediction_batch_size);
  ABSL_CHECK(options.prediction_batch_size % options.worker_threads == 0)
      << "batch size must be an integer multiple of #threads";

  EmbeddedTrainingServiceClient client(options.model_path);
  auto km = client.FetchLatestModel("");
  if (!km.ok()) {
    ABSL_LOG(ERROR) << "Failed to fetch model: " << km.status();
    return;
  }
  auto& [key, model] = *km;
  auto device = torch::kCPU;
  if (options.device == "mps") {
    device = torch::kMPS;
  } else if (options.device == "cuda") {
    device = torch::kCUDA;
  }
  model.to(device);

  ConcurrentQueue<FiberRequest> request_queue;

  std::thread gpu_pipeline_thread(GPUPipelineThread, std::ref(model),
                                  std::ref(request_queue), options);

  const int n_threads = options.worker_threads;
  const int fibers_per_thread = options.prediction_batch_size / n_threads;
  const int rounds_per_fiber = options.rounds;
  int64_t t_start = UnixMicros();
  std::vector<std::thread> fiber_threads;
  for (int k = 0; k < n_threads; k++) {
    fiber_threads.emplace_back([&] {
      std::vector<boost::fibers::fiber> fibers;
      for (int i = 0; i < fibers_per_thread; ++i) {
        int fiber_id = i;
        fibers.emplace_back(FiberWorker, k * n_threads + fiber_id,
                            std::ref(request_queue), rounds_per_fiber);
      }
      for (auto& fiber : fibers) {
        if (fiber.joinable()) {
          fiber.join();
        }
      }
    });
  }
  for (auto& t : fiber_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  int64_t t_end = UnixMicros();
  ABSL_LOG(INFO) << "Done with fibers";
  request_queue.push(FiberRequest{.done = true});
  if (gpu_pipeline_thread.joinable()) {
    gpu_pipeline_thread.join();
  }

  int final_op_count = options.rounds * options.prediction_batch_size;
  float ops_per_sec =
      static_cast<float>(final_op_count) / (t_end - t_start) * 1e6;
  ABSL_LOG(INFO) << "Performed " << final_op_count << " computations in "
                 << (t_end - t_start) / 1000 << "ms (" << ops_per_sec
                 << " ops/s). batch size:" << options.prediction_batch_size;
}

}  // namespace hexz

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  hexz::Options options = hexz::Options::FromFlags();
  std::string mode = absl::GetFlag(FLAGS_mode);
  // Run GPU performance test and exit.
  if (mode == "batch") {
    hexz::RunGPUBenchmarkBatch(options);
  } else if (mode == "task") {
    hexz::RunGPUBenchmarkMultiTasked(options);
  } else if (mode == "single") {
    hexz::RunGPUBenchmark(options);
  } else if (mode == "fiber") {
    hexz::RunGPUBenchmarkFiber(options);
  } else {
    ABSL_LOG(ERROR) << "Invalid mode: " << mode;
  }
  return 0;
}

/*
I1019 12:32:20.550976 1818297 gpubench_main.cc:520] batch_size=1: finished 1000
iterations in 3107ms (321.845 ops/s) I1019 12:32:24.097025 1818297
gpubench_main.cc:520] batch_size=2: finished 1000 iterations in 3166ms (631.643
ops/s) I1019 12:32:27.473312 1818297 gpubench_main.cc:520] batch_size=4:
finished 1000 iterations in 3014ms (1326.75 ops/s) I1019 12:32:31.608555 1818297
gpubench_main.cc:520] batch_size=8: finished 1000 iterations in 3707ms (2157.95
ops/s) I1019 12:32:36.850300 1818297 gpubench_main.cc:520] batch_size=16:
finished 1000 iterations in 4707ms (3399.03 ops/s) I1019 12:32:43.119426 1818297
gpubench_main.cc:520] batch_size=32: finished 1000 iterations in 5751ms (5564.24
ops/s) I1019 12:32:50.745617 1818297 gpubench_main.cc:520] batch_size=64:
finished 1000 iterations in 6930ms (9234.6 ops/s) I1019 12:33:01.024095 1818297
gpubench_main.cc:520] batch_size=128: finished 1000 iterations in 9284ms
(13786.7 ops/s)
*/