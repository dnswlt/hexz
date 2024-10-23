#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <boost/fiber/all.hpp>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "base.h"
#include "batch.h"
#include "hexz.pb.h"

namespace hexz {

// Interface class for a model than can make predictions for
// move likelihoods and board evaluations.
class Model {
 public:
  struct Prediction {
    torch::Tensor move_probs;
    float value;
  };
  // Updates the Model's underlying torch Module.
  virtual void UpdateModel(hexzpb::ModelKey key,
                           torch::jit::Module&& module) = 0;
  // Models tend to be used in a multi-threaded environment. Each thread must
  // register itself with the model by calling RegisterThread. The thread can
  // assume that all necessary cleanup/unregistration is done by the d'tor of
  // the returned ScopeGuard. This means that the Model MUST ONLY BE USED
  // while the ScopeGuard is in scope. A typical usage looks thus:
  //
  // {
  //   auto guard = model->RegisterThread();
  //   // ...
  //   auto prediction = model->Predict(board, node);
  //   // ...
  // }
  //
  // The ScopeGuard returned by the default implementation of RegisterThread
  // does nothing.
  virtual ScopeGuard RegisterThread() {
    return ScopeGuard([] {});
  }
  // This is the core method of any model, that returns a model prediction
  // for the given board and (MCTS search) node.
  virtual Prediction Predict(torch::Tensor board,
                             torch::Tensor action_mask) = 0;
  virtual hexzpb::ModelKey Key() const = 0;
  virtual ~Model() = default;
};

// Implementation of Model that uses an actual PyTorch ScriptModule.
// This implementation is synchronous, i.e. each call to Predict is
// immediately evaluated by the model.
class TorchModel : public Model {
 public:
  explicit TorchModel(torch::jit::Module&& module)
      : module_{std::move(module)} {}
  TorchModel(hexzpb::ModelKey key, torch::jit::Module&& module)
      : key_{std::move(key)}, module_{std::move(module)} {}

  void UpdateModel(hexzpb::ModelKey key, torch::jit::Module&& model) override;

  Prediction Predict(torch::Tensor board, torch::Tensor action_mask) override;

  // Returns the model key, if it was set during construction. Otherwise,
  // returns the "zero value" of ModelKey.
  hexzpb::ModelKey Key() const override { return key_; }
  // Sets the device on which model predictions will be made.
  // The default is torch::kCPU and does not need to be set explicitly.
  void SetDevice(torch::DeviceType device);

 private:
  hexzpb::ModelKey key_;
  torch::jit::Module module_;
  torch::DeviceType device_ = torch::kCPU;
};

// Implementation of Model that uses an actual PyTorch ScriptModule.
// This implementation uses a hexz::Batcher, batching model predictions
// across all threads working on independent MCTS search trees.
class BatchedTorchModel : public Model {
 public:
  class ComputeT {
   public:
    struct input_t {
      torch::Tensor board;
      torch::Tensor action_mask;
    };
    using result_t = Model::Prediction;
    ComputeT(torch::jit::Module&& module, torch::DeviceType device)
        : module_{std::move(module)}, device_{device} {
      module_.to(device);
    }
    std::vector<result_t> ComputeAll(std::vector<input_t>&& inputs);

   private:
    torch::jit::Module module_;
    torch::DeviceType device_;
  };
  BatchedTorchModel(hexzpb::ModelKey key, torch::jit::Module&& module,
                    torch::DeviceType device, int batch_size,
                    int64_t timeout_micros)
      : key_{key},
        device_{device},
        batcher_{std::make_unique<ComputeT>(std::move(module), device),
                 batch_size, timeout_micros} {}

  Prediction Predict(torch::Tensor board, torch::Tensor action_mask) override;

  // UpdateModel updates the model used
  // Access to this method must be synchronized across threads by callers!
  void UpdateModel(hexzpb::ModelKey key, torch::jit::Module&& module) override;

  // Returns the key of the currently used model.
  // Access to this method must be synchronized across threads by callers!
  hexzpb::ModelKey Key() const override;

  ScopeGuard RegisterThread() override { return batcher_.RegisterThread(); }

 private:
  hexzpb::ModelKey key_;
  torch::DeviceType device_;
  Batcher<ComputeT> batcher_;
};

// Implementation of Model that expects clients to run as independent fibers.
class FiberTorchModel : public Model {
 public:
  struct PredictionRequest {
    torch::Tensor board;
    torch::Tensor action_mask;
    // Promise to send the result back to the fiber.
    boost::fibers::promise<Model::Prediction> result_promise;
    // The final request sets this to true to terminate the GPU pipeline thread.
    bool done = false;
  };

  FiberTorchModel(hexzpb::ModelKey key, torch::jit::Module&& module,
                  torch::DeviceType device, int batch_size,
                  bool support_suspension);
  FiberTorchModel(const FiberTorchModel&) = delete;
  FiberTorchModel& operator=(const FiberTorchModel&) = delete;
  void UpdateModel(hexzpb::ModelKey key, torch::jit::Module&& model) override;

  Prediction Predict(torch::Tensor board, torch::Tensor action_mask) override;

  hexzpb::ModelKey Key() const override;

  // Each fiber using this model must call this method first and may only
  // use the model while the returned ScopeGuard is alive.
  // TODO: Rename to just Register, since here we don't expect threads, but
  // fibers?
  ScopeGuard RegisterThread() override;

  // Terminates the gpu_pipeline_thread_.
  ~FiberTorchModel();

  // When a worker runs on the same machine as the training server,
  // the Suspend and Resume methods can be used to keep the worker
  // from concurrently using the GPU.
  void Suspend();
  void Resume();

 private:
  // Executed by the gpu_pipeline_thread_, RunGPUPipeline
  // consumes requests from the request_queue_, and sends them
  // to the GPU in batches.
  // The GPU pipeline thread is terminated when there are no fibers left.
  void RunGPUPipeline();
  // Called by the ScopeGuard returned by RegisterThread. Notifies
  // the GPU pipeline thread.
  void Unregister();

  // Moves up to max_batch_size_ elements from the request queue into buf.
  // This method takes the number of active threads into account and
  // blocks until data for the maximum allowed batch size, i.e.
  // min(active_fibers, max_batch_size_) have been read.
  // It returns the number of items moved into batch. This can only
  // be zero if there are no more active fibers.
  int ReadBatch(std::vector<PredictionRequest>& batch);
  // Adds the given request to the queue, for consumption by the GPU pipeline.
  void PushRequest(PredictionRequest&& request);

  // Mutex for access to the request queue and associated cv and counters.
  mutable std::mutex request_mut_;
  std::thread gpu_pipeline_thread_;
  std::queue<PredictionRequest> request_queue_;
  // Condition variable to notify the GPU pipeline thread.
  std::condition_variable request_queue_cv_;
  int active_fibers_ = 0;
  // Used to signal that a fiber has left the building.
  bool fiber_left_ = false;
  const bool support_suspension_;
  bool suspended_ = false;
  // CV for suspending the GPU pipeline thread.
  std::condition_variable suspension_cv_;

  // Mutex for the access to the model.
  mutable std::mutex module_mut_;
  torch::jit::Module module_;
  hexzpb::ModelKey key_;
  const int max_batch_size_;
  const torch::DeviceType device_;
};

}  // namespace hexz
