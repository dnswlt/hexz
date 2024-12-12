#include "model.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <boost/fiber/barrier.hpp>

#include "board.h"

namespace hexz {

namespace {

torch::jit::Module LoadModule() {
  auto scriptmodule = torch::jit::load("testdata/scriptmodule.pt", torch::kCPU);
  scriptmodule.eval();
  return scriptmodule;
}

torch::jit::Module LoadModuleRes10(torch::DeviceType device) {
  auto scriptmodule =
      torch::jit::load("testdata/scriptmodule_res10.pt", device);
  scriptmodule.eval();
  return scriptmodule;
}

torch::Tensor ActionMask(const Board& board, int player) {
  auto moves = board.NextMoves(player);
  auto action_mask =
      torch::zeros({2, 11, 10}, c10::TensorOptions().dtype(torch::kBool));
  auto action_mask_acc = action_mask.accessor<bool, 3>();
  for (const auto& m : moves) {
    action_mask_acc[static_cast<size_t>(m.typ)][m.r][m.c] = true;
  }
  return action_mask;
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

TEST(TorchModelTest, Deterministic) {
  // Checks that model predictions are deterministic.
  hexzpb::ModelKey dummy_key;
  TorchModel model(dummy_key, LoadModule(), torch::kCPU);
  // Prepare inputs.
  auto board = torch::randn({11, 11, 10});
  auto action_mask = torch::rand({2, 11, 10}) < 0.5;

  auto pred_0 = model.Predict(board, action_mask);
  for (int i = 0; i < 10; i++) {
    auto pred = model.Predict(board, action_mask);
    EXPECT_FLOAT_EQ(pred_0.value, pred.value);
    EXPECT_TRUE(pred_0.move_probs.equal(pred.move_probs));
  }
}

TEST(BatchedTorchModelTest, SmokeTestSingleThreaded) {
  auto scriptmodule = LoadModule();
  // Happy path for BatchedTorchModel.
  constexpr int batch_size = 4;
  constexpr int64_t timeout_micros = 1'000'000;
  BatchedTorchModel m(hexzpb::ModelKey(), std::move(scriptmodule), torch::kCPU,
                      batch_size, timeout_micros);
  auto token = m.Enter();
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
      auto token = m.Enter();
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
  { auto token = model.Enter(); }
}

TEST(FiberTorchModelTest, SmokeTestSingleFiber) {
  auto scriptmodule = LoadModule();
  // Even if the batch size is large, a single fiber should be able
  // to retrieve a result, since the GPU pipeline thread keeps track
  // of the number of active fibers.
  const int batch_size = 16;
  FiberTorchModel model(hexzpb::ModelKey(), std::move(scriptmodule),
                        torch::kCPU, batch_size, false);
  auto token = model.Enter();
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
        // Not calling Enter here. This should lead to a failure.
        // auto token = model.Enter();
        auto board = torch::randn({11, 11, 10});
        auto action_mask = torch::rand({2, 11, 10}) < 0.5;
        auto pred = model.Predict(board, action_mask);
      },
      "Enter");
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
          auto token = model.Enter();
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

testing::AssertionResult TensorsApproxEq(torch::Tensor t1, torch::Tensor t2,
                                         float eps_max, float eps_mean) {
  auto d = torch::abs(t1 - t2);
  float max_d = torch::max(d).item<float>();
  auto nonzero_values = d.masked_select(torch::logical_or(t1 != 0, t2 != 0));
  float mean_d = 0.0;
  if (nonzero_values.numel() > 0) {
    mean_d = torch::mean(nonzero_values).item<float>();
  }
  if (max_d < eps_max && mean_d < eps_mean) {
    return testing::AssertionSuccess();
  }

  return testing::AssertionFailure()
         << "max delta is " << max_d << " (mean: " << mean_d
         << ") for tensors (*100) " << (t1 * 100) << "\n=== VS ===\n"
         << (t2 * 100);
}

TEST(FiberTorchModelTest, ConcurrentResultsDeviceCorrect) {
  // Tests that FiberTorchModel's concurrent results are approx. equal
  // to those we get from a sequential TorchModel.

  // We cannot fully enable deterministic algorithms, since our network uses
  // CUBLAS
  // https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
  //   at::globalContext().setDeterministicCuDNN(true);
  //   at::globalContext().setDeterministicAlgorithms(true,
  //   /*warn_only=*/false);
  torch::DeviceType device;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  } else if (torch::mps::is_available()) {
    device = torch::kMPS;
  } else {
    // We could also run the test on the CPU, but I'd rather see a skip
    // warning than silently run it on the CPU, which is a non-use case.
    GTEST_SKIP() << "Skipping test: no applicable device available (cuda, mps)";
  }
  hexzpb::ModelKey dummy_key;
  TorchModel torch_model(dummy_key, LoadModuleRes10(device), device);
  // Prepare inputs.
  const int N = 128;
  // boards is populated with valid random boards.
  auto boards = torch::zeros({N, 11, 11, 10});
  auto action_masks =
      torch::zeros({N, 2, 11, 10}, c10::TensorOptions().dtype(torch::kBool));
  for (int i = 0; i < N; i++) {
    int player = 0;
    Board board = Board::RandomBoard();
    boards[i] = board.Tensor(player);
    action_masks[i] = ActionMask(board, player);
  }

  // Compute sequential predictions.
  std::vector<hexz::Model::Prediction> seq_preds;
  for (int i = 0; i < N; i++) {
    auto pred = torch_model.Predict(boards[i], action_masks[i]);
    seq_preds.push_back(pred);
  }

  // Compute concurrent predictions.
  const int n_threads = 2;
  const int fibers_per_thread = N / n_threads;
  const int batch_size = 2 * fibers_per_thread;
  FiberTorchModel fiber_torch_model(hexzpb::ModelKey(), LoadModuleRes10(device),
                                    device, batch_size, false);
  std::mutex mut;
  std::vector<std::thread> threads;
  std::vector<hexz::Model::Prediction> conc_preds(n_threads *
                                                  fibers_per_thread);
  boost::fibers::barrier barrier(n_threads * fibers_per_thread);
  int idx = 0;
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([&, thread_start_idx = idx] {
      std::vector<boost::fibers::fiber> fibers;
      for (int j = 0; j < fibers_per_thread; j++) {
        fibers.emplace_back([&, fiber_idx = thread_start_idx + j] {
          barrier.wait();
          auto token = fiber_torch_model.Enter();
          auto pred = fiber_torch_model.Predict(boards[fiber_idx],
                                                action_masks[fiber_idx]);
          {
            std::scoped_lock<std::mutex> lk(mut);
            conc_preds[fiber_idx] = pred;
          }
        });
      }
      std::for_each(fibers.begin(), fibers.end(), [](auto& f) { f.join(); });
    });
    idx += fibers_per_thread;
  }
  std::for_each(threads.begin(), threads.end(), [](auto& t) { t.join(); });

  ASSERT_EQ(seq_preds.size(), conc_preds.size());

  // The results are almost shockingly coarsely equal. eps == 1e-4 routinely
  // fails. So instead, we reuqire a max 0.5 % points absolute deviation,
  // and a mean 0.01 % points absolute deviation (among non-zero entries).
  const float eps_max = 0.005;
  const float eps_mean = 0.0001;
  for (int i = 0; i < seq_preds.size(); i++) {
    EXPECT_LT(std::abs(seq_preds[i].value - conc_preds[i].value), eps_max);
    ASSERT_TRUE(TensorsApproxEq(seq_preds[i].move_probs,
                                conc_preds[i].move_probs, eps_max, eps_mean));
  }
}

}  // namespace
}  // namespace hexz
