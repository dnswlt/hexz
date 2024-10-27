#include "model.h"

#include <absl/log/absl_log.h>

#include "perfm.h"

namespace hexz {

void TorchModel::UpdateModel(hexzpb::ModelKey key,
                             torch::jit::Module&& module) {
  key_ = std::move(key);
  module_ = std::move(module);
  module_.to(device_);
}

Model::Prediction TorchModel::Predict(torch::Tensor board,
                                      torch::Tensor action_mask) {
  Perfm::Scope ps(Perfm::Predict);
  torch::NoGradGuard no_grad;
  auto board_batch = board.unsqueeze(0).to(device_);
  auto action_mask_batch = action_mask.flatten().unsqueeze(0).to(device_);
  std::vector<torch::jit::IValue> inputs{
      board_batch,
      action_mask_batch,
  };
  auto output = module_.forward(inputs);

  // The model should output two values: the move likelihoods as a [1, 220]
  // tensor of logits (1 being the batch size), and a single float value
  // prediction.
  ABSL_DCHECK(output.isTuple());
  const auto output_tuple = output.toTuple();
  ABSL_DCHECK(output_tuple->size() == 2);
  // Read logits back to CPU / main memory.
  const auto logits = output_tuple->elements()[0].toTensor().to(torch::kCPU);
  const auto dim = logits.sizes();
  ABSL_DCHECK(dim.size() == 2 && dim[0] == 1 && dim[1] == 2 * 11 * 10);
  const auto value = output_tuple->elements()[1].toTensor().item<float>();
  return Prediction{
      .move_probs = logits.reshape({2, 11, 10}).exp(),
      .value = value,
  };
}

std::vector<Model::Prediction> BatchedTorchModel::ComputeT::ComputeAll(
    std::vector<ComputeT::input_t>&& inputs) {
  Perfm::Scope ps(Perfm::PredictBatch);
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
  ABSL_DCHECK(pred.isTuple());
  const auto output_tuple = pred.toTuple();
  ABSL_DCHECK(output_tuple->size() == 2);
  // Read logits back to CPU / main memory.
  const auto probs =
      output_tuple->elements()[0].toTensor().exp().to(torch::kCPU);
  const auto dim = probs.sizes();
  ABSL_DCHECK(dim.size() == 2 && dim[0] == n_inputs && dim[1] == 2 * 11 * 10);
  const auto values = output_tuple->elements()[1].toTensor().to(torch::kCPU);
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

Model::Prediction BatchedTorchModel::Predict(torch::Tensor board,
                                             torch::Tensor action_mask) {
  Perfm::Scope ps(Perfm::Predict);
  return batcher_.ComputeValue(ComputeT::input_t{
      .board = board,
      .action_mask = action_mask.flatten(),
  });
}

void BatchedTorchModel::UpdateModel(hexzpb::ModelKey key,
                                    torch::jit::Module&& module) {
  key_ = std::move(key);
  batcher_.UpdateComputeT(std::make_unique<BatchedTorchModel::ComputeT>(
      std::move(module), device_));
}

hexzpb::ModelKey BatchedTorchModel::Key() const { return key_; }

FiberTorchModel::FiberTorchModel(hexzpb::ModelKey key,
                                 torch::jit::Module&& module,
                                 torch::DeviceType device, int batch_size,
                                 bool support_suspension)
    : key_{std::move(key)},
      module_{std::move(module)},
      device_{device},
      max_batch_size_{batch_size},
      support_suspension_{support_suspension} {
  // Acquire lock to have a synchronization point for module_ before any access
  // to it by other threads or fibers. (Necessary?)
  std::scoped_lock<std::mutex> lk(module_mut_);
  module_.to(device_);
  // Start the GPU thread only after this object has been
  // fully initialized.
  gpu_pipeline_thread_ = std::thread{&FiberTorchModel::RunGPUPipeline, this};
}

FiberTorchModel::~FiberTorchModel() {
  {
    std::scoped_lock<std::mutex> lk(request_mut_);
    ABSL_CHECK(active_fibers_ == 0)
        << "FiberTorchModel is being destroyed while fibers are active";
  }
  if (gpu_pipeline_thread_.joinable()) {
    gpu_pipeline_thread_.join();
  }
}

void FiberTorchModel::UpdateModel(hexzpb::ModelKey key,
                                  torch::jit::Module&& module) {
  std::scoped_lock<std::mutex> lk(module_mut_);
  key_ = std::move(key);
  module_ = std::move(module);
  module_.to(device_);
}

hexzpb::ModelKey FiberTorchModel::Key() const {
  std::scoped_lock<std::mutex> lk(module_mut_);
  return key_;
}

Model::Prediction FiberTorchModel::Predict(torch::Tensor board,
                                           torch::Tensor action_mask) {
  Perfm::Scope ps(Perfm::Predict);
  boost::fibers::promise<Model::Prediction> promise;
  auto future = promise.get_future();
  PushRequest(PredictionRequest{
      .board = board,
      .action_mask = action_mask.flatten(),
      .result_promise = std::move(promise),
  });
  return future.get();
}

void FiberTorchModel::PushRequest(PredictionRequest&& request) {
  std::scoped_lock<std::mutex> lk(request_mut_);
  ABSL_CHECK(active_fibers_ > 0)
      << "Called PushRequest with zero active fibers. Must call RegisterThread "
         "for each fiber first.";
  request_queue_.push(std::move(request));
  request_queue_cv_.notify_one();
}

int FiberTorchModel::ReadBatch(std::vector<PredictionRequest>& batch) {
  std::unique_lock<std::mutex> lock(request_mut_);
  if (support_suspension_) {
    suspension_cv_.wait(lock, [this] { return !suspended_; });
  }

  int batch_size = std::min(active_fibers_, max_batch_size_);
  int k = 0;
  while (batch.size() < batch_size) {
    request_queue_cv_.wait(
        lock, [this] { return !request_queue_.empty() || fiber_left_; });
    if (fiber_left_) {
      fiber_left_ = false;
      if (active_fibers_ < batch_size) {
        batch_size = active_fibers_;
      }
    }
    while (batch.size() < batch_size && !request_queue_.empty()) {
      batch.push_back(std::move(request_queue_.front()));
      request_queue_.pop();
      k++;
    }
  }
  return k;
}

void FiberTorchModel::RunGPUPipeline() {
  // Ensure this thread's perfm stats are accumulated into total stats.
  Perfm::ThreadScope perfm_thread_scope;
  ABSL_LOG(INFO) << "FiberTorchModel::GPUPipeline started";
  std::vector<PredictionRequest> batch;
  batch.reserve(max_batch_size_);
  {
    // Wait for initial fiber to join
    std::unique_lock<std::mutex> init_lk(request_mut_);
    request_queue_cv_.wait(
        init_lk, [this] { return active_fibers_ > 0 || fiber_left_; });
    if (fiber_left_) {
      ABSL_LOG(ERROR) << "Fiber registered and unregistered from GPU pipeline "
                         "without ever calling Predict. Terminating.";
      return;
    }
  }
  while (true) {
    // Collect requests for batching
    ReadBatch(batch);
    if (batch.empty()) {
      ABSL_LOG(INFO) << "Nothing left to read from the queue. Exiting.";
      return;
    }
    const size_t n_inputs = batch.size();
    std::vector<torch::Tensor> boards;
    boards.reserve(n_inputs);
    std::vector<torch::Tensor> action_masks;
    action_masks.reserve(n_inputs);
    for (auto& req : batch) {
      boards.emplace_back(std::move(req.board));
      action_masks.emplace_back(std::move(req.action_mask));
    }

    std::vector<torch::jit::IValue> model_inputs = {
        torch::stack(boards).to(device_),
        torch::stack(action_masks).to(device_),
    };
    {
      Perfm::Scope perfm(Perfm::PredictBatch);
      // Need to guard access to module_ b/c it might get updated concurrently.
      std::unique_lock<std::mutex> lk(module_mut_);
      auto pred = module_.forward(model_inputs);
      lk.unlock();
      ABSL_DCHECK(pred.isTuple());
      const auto output_tuple = pred.toTuple();
      ABSL_DCHECK(output_tuple->size() == 2);
      // Read logits back to CPU / main memory.
      const auto probs =
          output_tuple->elements()[0].toTensor().exp().to(torch::kCPU);
      const auto dim = probs.sizes();
      ABSL_DCHECK(dim.size() == 2 && dim[0] == n_inputs &&
                  dim[1] == 2 * 11 * 10);
      const auto values =
          output_tuple->elements()[1].toTensor().to(torch::kCPU);
      for (int i = 0; i < n_inputs; i++) {
        batch[i].result_promise.set_value(Model::Prediction{
            .move_probs = probs.index({i}).reshape({2, 11, 10}),
            .value = values.index({i}).item<float>(),
        });
      }
      batch.clear();
    }
  }
}

ScopeGuard FiberTorchModel::RegisterThread() {
  std::scoped_lock<std::mutex> lk(request_mut_);
  active_fibers_++;
  request_queue_cv_.notify_one();
  return ScopeGuard([this] { Unregister(); });
}

void FiberTorchModel::Unregister() {
  std::scoped_lock<std::mutex> lk(request_mut_);
  ABSL_CHECK(active_fibers_ > 0);
  active_fibers_--;
  fiber_left_ = true;
  request_queue_cv_.notify_one();
}

void FiberTorchModel::Suspend() {
  ABSL_CHECK(support_suspension_) << "Suspension not enabled";
  ABSL_LOG(INFO) << "Suspending GPU pipeline thread";
  std::scoped_lock lk(request_mut_);
  suspended_ = true;
}
void FiberTorchModel::Resume() {
  ABSL_CHECK(support_suspension_) << "Suspension not enabled";
  ABSL_LOG(INFO) << "Resuming GPU pipeline thread";
  std::scoped_lock lk(request_mut_);
  suspended_ = false;
  suspension_cv_.notify_one();
}
}  // namespace hexz
