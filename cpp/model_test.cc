#include "model.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace hexz {

namespace {

torch::jit::Module LoadModule() {
  auto scriptmodule = torch::jit::load("testdata/scriptmodule.pt");
  scriptmodule.to(torch::kCPU);
  scriptmodule.eval();
  return scriptmodule;
}

TEST(PredictBatchTest, Shape) {
  auto m = LoadModule();
  auto boards = torch::rand({5, 11, 11, 10});
  auto action_masks = torch::rand({5, 2, 11, 10}) < 0.5;
  std::vector<torch::jit::IValue> inputs{
      boards,
      action_masks,
  };
  BatchPrediction pred = PredictBatch(m, std::move(inputs));
  EXPECT_THAT(pred.policy.sizes(), testing::ElementsAre(5, 2, 11, 10));
  EXPECT_THAT(pred.values.sizes(), testing::ElementsAre(5, 1));
  // Sum each policy in the batch and copy the result into a vector.
  auto policy_sum = pred.policy.view({pred.policy.size(0), -1}).sum(1);
  std::vector<float> sums(policy_sum.data_ptr<float>(),
                          policy_sum.data_ptr<float>() + policy_sum.size(0));
  // Copy values to vector.
  std::vector<float> values(
      pred.values.data_ptr<float>(),
      pred.values.data_ptr<float>() + pred.values.size(0));
  // Policies should sum to 1, since they are interpreted as probabilities.
  EXPECT_THAT(sums, testing::Each(testing::FloatNear(1, 1e-2)));
  // values should be in [-1, 1].
  EXPECT_THAT(values, testing::Each(testing::Ge(-1)));
  EXPECT_THAT(values, testing::Each(testing::Le(1)));
}

TEST(BatchedTorchModelTest, SmokeTestSingleThreaded) {
  auto scriptmodule = LoadModule();
  // Happy path for BatchedTorchModel.
  constexpr int batch_size = 4;
  constexpr int64_t timeout_micros = 1'000'000;
  BatchedTorchModel m(hexzpb::ModelKey(), std::move(scriptmodule), torch::kCPU,
                      batch_size, timeout_micros);
  auto token = m.RegisterThread();
  // Prepare inputs.
  auto board = torch::randn({11, 11, 10});
  auto action_mask = torch::rand({2, 11, 10}) < 0.5;

  // Execute.
  auto pred = m.Predict(board, action_mask);

  // Validate.
  auto sizes = pred.move_probs.sizes();
  EXPECT_THAT(sizes, testing::ElementsAre(2, 11, 10));
  EXPECT_TRUE(std::abs(pred.move_probs.sum().item<float>() - 1.0) < 0.01);
}

TEST(BatchedTorchModelTest, SmokeTestMultiThreaded) {
  auto scriptmodule = LoadModule();
  constexpr int batch_size = 8;
  constexpr int64_t timeout_micros = 1'000'000;
  BatchedTorchModel m(hexzpb::ModelKey(), std::move(scriptmodule), torch::kCPU,
                      batch_size, timeout_micros);
  std::vector<std::thread> ts(batch_size);
  std::vector<float> sum_pr(ts.size(), 0);
  std::mutex mut;
  for (int i = 0; i < batch_size; i++) {
    ts[i] = std::thread([&, i] {
      auto token = m.RegisterThread();
      // Prepare inputs.
      auto board = torch::randn({11, 11, 10});
      auto action_mask = torch::rand({2, 11, 10}) < 0.5;

      // Execute.
      auto pred = m.Predict(board, action_mask);

      // Record results.
      {
        std::scoped_lock<std::mutex> lk(mut);
        sum_pr[i] = pred.move_probs.sum().item<float>();
      }
    });
  }
  std::for_each(ts.begin(), ts.end(), [](auto& t) { t.join(); });
  EXPECT_THAT(sum_pr, testing::Each(testing::FloatNear(1, 1e-2)));
}

TEST(FiberTorchModelTest, FiberTorchModelRegisterUnregister) {
  // Register a fiber, but never make any calls. GPU pipeline thread should shut
  // down cleanly.
  auto scriptmodule = LoadModule();
  const int batch_size = 1;
  FiberTorchModel model(hexzpb::ModelKey(), std::move(scriptmodule),
                        torch::kCPU, batch_size, false);
  { auto token = model.RegisterThread(); }
}

TEST(FiberTorchModelTest, SmokeTestSingleFiber) {
  auto scriptmodule = LoadModule();
  // Even if the batch size is large, a single fiber should be able
  // to retrieve a result, since the GPU pipeline thread keeps track
  // of the number of active fibers.
  const int batch_size = 16;
  FiberTorchModel model(hexzpb::ModelKey(), std::move(scriptmodule),
                        torch::kCPU, batch_size, false);
  auto token = model.RegisterThread();
  auto board = torch::randn({11, 11, 10});
  auto action_mask = torch::rand({2, 11, 10}) < 0.5;
  auto pred = model.Predict(board, action_mask);
  // Validate.
  auto sizes = pred.move_probs.sizes();
  EXPECT_THAT(sizes, testing::ElementsAre(2, 11, 10));
  EXPECT_TRUE(std::abs(pred.move_probs.sum().item<float>() - 1.0) < 0.01);
}

TEST(FiberTorchModelDeathTest, CheckFailIfNotRegistered) {
  ASSERT_DEATH(
      {
        auto scriptmodule = LoadModule();
        const int batch_size = 1;
        FiberTorchModel model(hexzpb::ModelKey(), std::move(scriptmodule),
                              torch::kCPU, batch_size, false);
        // Not calling RegisterThread here. This should lead to a failure.
        // auto token = model.RegisterThread();
        auto board = torch::randn({11, 11, 10});
        auto action_mask = torch::rand({2, 11, 10}) < 0.5;
        auto pred = model.Predict(board, action_mask);
      },
      "RegisterThread");
}

TEST(FiberTorchModelTest, SmokeTestMultipleFibers) {
  auto scriptmodule = LoadModule();
  const int batch_size = 4;
  const int n_threads = 2;
  const int fibers_per_thread = 4;
  FiberTorchModel model(hexzpb::ModelKey(), std::move(scriptmodule),
                        torch::kCPU, batch_size, false);
  std::vector<float> sum_pr;
  std::mutex mut;
  std::vector<std::thread> threads;
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([&] {
      std::vector<boost::fibers::fiber> fibers;
      for (int j = 0; j < fibers_per_thread; j++) {
        fibers.emplace_back([&] {
          auto token = model.RegisterThread();
          auto board = torch::randn({11, 11, 10});
          auto action_mask = torch::rand({2, 11, 10}) < 0.5;
          auto pred = model.Predict(board, action_mask);
          {
            std::scoped_lock<std::mutex> lk(mut);
            sum_pr.push_back(pred.move_probs.sum().item<float>());
          }
        });
      }
      std::for_each(fibers.begin(), fibers.end(), [](auto& f) { f.join(); });
    });
  }
  std::for_each(threads.begin(), threads.end(), [](auto& t) { t.join(); });
  // Validate.
  EXPECT_THAT(sum_pr, testing::Each(testing::FloatNear(1, 1e-2)));
}

}  // namespace
}  // namespace hexz
