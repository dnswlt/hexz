#include "model.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace hexz {

namespace {

TEST(FiberTorchModelTest, SmokeTestSingleFiber) {
  auto scriptmodule = torch::jit::load("testdata/scriptmodule.pt");
  scriptmodule.to(torch::kCPU);
  scriptmodule.eval();
  const int batch_size = 1;
  FiberTorchModel model(hexzpb::ModelKey(), std::move(scriptmodule),
                        torch::kCPU, batch_size);
  auto token = model.RegisterThread();
  auto board = torch::randn({11, 11, 10});
  auto action_mask = torch::rand({2, 11, 10}) < 0.5;
  auto pred = model.Predict(board, action_mask);
  // Validate.
  auto sizes = pred.move_probs.sizes();
  EXPECT_THAT(sizes, testing::ElementsAre(2, 11, 10));
  EXPECT_TRUE(std::abs(pred.move_probs.sum().item<float>() - 1.0) < 0.01);
}

TEST(FiberTorchModelTest, SmokeTestMultipleFibers) {
  auto scriptmodule = torch::jit::load("testdata/scriptmodule.pt");
  scriptmodule.to(torch::kCPU);
  scriptmodule.eval();
  const int batch_size = 4;
  const int n_threads = 2;
  const int fibers_per_thread = 4;
  FiberTorchModel model(hexzpb::ModelKey(), std::move(scriptmodule),
                        torch::kCPU, batch_size);
  std::vector<float> sum_pr;
  std::mutex mut;
  std::vector<std::thread> threads;
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([&, i] {
      std::vector<boost::fibers::fiber> fibers;
      for (int j = 0; j < fibers_per_thread; j++) {
        fibers.emplace_back([&, j] {
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
