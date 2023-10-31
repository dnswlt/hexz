#include "mcts.h"

#include <gtest/gtest.h>

#include <vector>

namespace hexz {

std::mt19937 rng{std::random_device{}()};

TEST(MCTSTest, TensorAsVector) {
  torch::Tensor t = torch::rand({2, 11, 10}, torch::kFloat32);
  std::vector<float> data(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
  EXPECT_EQ(t.index({1, 7, 3}).item<float>(), data[1 * 11 * 10 + 7 * 10 + 3]);
}

}  // namespace hexz
