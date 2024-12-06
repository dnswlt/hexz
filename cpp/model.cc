#include "model.h"

#include <absl/log/absl_log.h>

#include "perfm.h"

namespace hexz {

BatchPrediction PredictBatch(torch::jit::Module& module,
                             std::vector<torch::jit::IValue>&& inputs) {
  auto pred = module.forward(std::move(inputs)).toTuple();
  const auto policy = torch::softmax(pred->elements()[0].toTensor(), 1)
                          .to(torch::kCPU)
                          .reshape({-1, 2, 11, 10});
  const auto values = pred->elements()[1].toTensor().to(torch::kCPU);
  return BatchPrediction{
      .policy = policy,
      .values = values,
  };
}

void TorchModel::UpdateModel(hexzpb::ModelKey key,
                             torch::jit::Module&& module) {
  key_ = std::move(key);
  module_ = std::move(module);
  module_.eval();
  module_.to(device_);
}

Model::Prediction TorchModel::Predict(torch::Tensor board,
                                      torch::Tensor action_mask) {
  Perfm::Scope ps(Perfm::Predict);
  auto board_batch = board.unsqueeze(0).to(device_);
  auto action_mask_batch = action_mask.unsqueeze(0).to(device_);
  std::vector<torch::jit::IValue> inputs{
      board_batch,
      action_mask_batch,
  };
  BatchPrediction pred = PredictBatch(module_, std::move(inputs));
  return Prediction{
      .move_probs = pred.policy.index({0}).reshape({2, 11, 10}),
      .value = pred.values.index({0}).item<float>(),
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
  auto pred = PredictBatch(module_, std::move(model_inputs));
  std::vector<Model::Prediction> result;
  result.reserve(n_inputs);
  for (int i = 0; i < n_inputs; i++) {
    result.push_back(Model::Prediction{
        .move_probs = pred.policy.index({i}),
        .value = pred.values.index({i}).item<float>(),
    });
  }
  return result;
}

Model::Prediction BatchedTorchModel::Predict(torch::Tensor board,
                                             torch::Tensor action_mask) {
  Perfm::Scope ps(Perfm::Predict);
  return batcher_.ComputeValue(ComputeT::input_t{
      .board = board,
      .action_mask = action_mask,
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
    : support_suspension_{support_suspension},
      module_{std::move(module)},
      key_{std::move(key)},
      max_batch_size_{batch_size},
      device_{device} {
  // Acquire lock to have a synchronization point for module_ before any access
  // to it by other threads or fibers. (Necessary?)
  std::scoped_lock<std::mutex> lk(module_mut_);
  module_.eval();
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
    shutdown_ = true;
    request_queue_cv_.notify_all();
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
  module_.eval();
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
      .action_mask = action_mask,
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
    request_queue_cv_.wait(lock, [this] {
      return !request_queue_.empty() || fiber_left_ || shutdown_;
    });
    if (fiber_left_) {
      fiber_left_ = false;  // acknowledge receipt of "fiber left" condition.
      if (active_fibers_ < batch_size) {
        batch_size = active_fibers_;
      }
    }
    if (shutdown_) {
      return k;  // We're done. Return last batch, if any.
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
  // Don't calculate gradients: self-play only needs model evaluation.
  torch::NoGradGuard no_grad;
  // Ensure this thread's perfm stats are accumulated into total stats.
  Perfm::ThreadScope perfm_thread_scope;
  ABSL_LOG(INFO) << "FiberTorchModel::GPUPipeline started";
  std::vector<PredictionRequest> batch;
  batch.reserve(max_batch_size_);
  while (true) {
    // Collect requests for batching
    while (ReadBatch(batch) == 0) {
      // If ReadBatch returns zero, either all active fibers left or we're
      // shutting down entirely. In the former case, we'll wait for new
      // fibers to arrive. In the latter case, we quit.
      std::unique_lock<std::mutex> lk(request_mut_);
      request_queue_cv_.wait(
          lk, [this] { return active_fibers_ > 0 || shutdown_; });
      if (shutdown_) {
        ABSL_LOG(INFO) << "GPU pipeline is shutting down.";
        return;
      }
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
      BatchPrediction pred = PredictBatch(module_, std::move(model_inputs));
      lk.unlock();
      auto policies = pred.policy.unbind(0);
      auto values = pred.values.unbind(0);
      for (int i = 0; i < n_inputs; i++) {
        batch[i].result_promise.set_value(Model::Prediction{
            .move_probs = policies[i],
            .value = values[i].item<float>(),
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

YarnTorchModel::YarnTorchModel(hexzpb::ModelKey key,
                               torch::jit::Module&& module,
                               torch::DeviceType device)
    : module_{std::move(module)}, key_{key}, device_{device} {}

YarnTorchModel::~YarnTorchModel() {
  ABSL_CHECK(active_fibers_ == 0)
      << "YarnTorchModel destroyed while fibers were active.";
}

ScopeGuard YarnTorchModel::RegisterThread() {
  active_fibers_++;
  return ScopeGuard([this] { Unregister(); });
}

void YarnTorchModel::Unregister() {
  active_fibers_--;
  if (active_fibers_ > 0 && active_fibers_ == waiting_) {
    // All other fibers are waiting for a computation.
    force_compute_ = true;
    prediction_queue_cv_.notify_one();
  }
}

void YarnTorchModel::UpdateModel(hexzpb::ModelKey key,
                                 torch::jit::Module&& model) {
  std::scoped_lock<std::mutex> lk(module_mut_);
  key_ = std::move(key);
  module_ = std::move(model);
  module_.to(device_);
  module_.eval();
}

hexzpb::ModelKey YarnTorchModel::Key() const {
  std::scoped_lock<std::mutex> lk(module_mut_);
  return key_;
}

void YarnTorchModel::ClearBuffers() {
  boards_.clear();
  action_masks_.clear();
  predictions_.clear();
}

void YarnTorchModel::ComputePredictions() {
  // All fibers have made their request. Let's predict!
  std::vector<torch::jit::IValue> model_inputs = {
      torch::stack(boards_).to(device_),
      torch::stack(action_masks_).to(device_),
  };
  // Model access needs to be synchronized across all threads.
  // (a) other threads also run predictions and (b) it may get updated.
  std::unique_lock<std::mutex> lk(module_mut_);
  BatchPrediction pred = PredictBatch(module_, std::move(model_inputs));
  lk.unlock();
  // Dispatch prediction results into result slots of all involved fibers.
  auto policies = pred.policy.unbind(0);
  auto values = pred.values.unbind(0);
  for (int i = 0; i < boards_.size(); i++) {
    predictions_.push_back(Model::Prediction{
        .move_probs = policies[i],
        .value = values[i].item<float>(),
    });
  }
  results_ready_ = boards_.size();
}

Model::Prediction YarnTorchModel::Predict(torch::Tensor board,
                                          torch::Tensor action_mask) {
  {
    // Wait until next batch round opens.
    std::unique_lock<boost::fibers::mutex> enter_lk(enter_mut_);
    enter_cv_.wait(enter_lk, [this]() { return results_ready_ == 0; });
  }
  boards_.push_back(board);
  action_masks_.push_back(action_mask);
  int data_idx = boards_.size() - 1;
  if (boards_.size() < active_fibers_) {
    // There is space left, just enqueue and wait for computation.
    std::unique_lock<boost::fibers::mutex> lk(prediction_queue_mut_);
    waiting_++;
    prediction_queue_cv_.wait(
        lk, [this]() { return results_ready_ > 0 || force_compute_; });
    if (force_compute_) {
      // We were woken up b/c a fiber left. Need to trigger a computation.
      ComputePredictions();
      force_compute_ = false;
      waiting_ = 0;
      prediction_queue_cv_.notify_all();
    }
    // Results are available.
    auto result = predictions_[data_idx];
    results_ready_--;
    if (results_ready_ == 0) {
      // This is the last fiber to retrieve its results. Start next round.
      ClearBuffers();
      lk.unlock();  // avoid pessimization.
      enter_cv_.notify_all();
    }
    return result;
  }
  // All fibers have made their request. Let's predict!
  ComputePredictions();
  // Grab result for this fiber.
  auto result = predictions_.back();
  results_ready_--;
  if (results_ready_ == 0) {
    // This fiber was the only one in the batch.
    ClearBuffers();
  } else {
    // Tell other fibers that data is available.
    waiting_ = 0;
    prediction_queue_cv_.notify_all();
  }
  return result;
}

}  // namespace hexz
