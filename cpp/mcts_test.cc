#include "mcts.h"

#include <absl/status/status.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <fstream>
#include <iterator>
#include <vector>

#include "hexz.pb.h"

namespace hexz {
namespace {

std::mt19937 rng{std::random_device{}()};

TEST(MCTSTest, TensorAsVector) {
  torch::Tensor t = torch::rand({2, 11, 10}, torch::kFloat32);
  std::vector<float> data(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
  EXPECT_EQ(t.index({1, 7, 3}).item<float>(), data[1 * 11 * 10 + 7 * 10 + 3]);
}

absl::Status ReadFile(const std::string& path, std::vector<char>& buf) {
  std::ifstream in(path, std::ios::binary);
  if (in.fail()) {
    return absl::NotFoundError("Failed to open file");
  }
  in.seekg(0, std::ios::end);
  buf.reserve(in.tellg());
  in.seekg(0, std::ios::beg);
  buf.assign(std::istreambuf_iterator<char>(in),
             std::istreambuf_iterator<char>());
  return absl::OkStatus();
}

TEST(MCTSTest, TorchPickleLoad) {
  // Load a (2, 2) Tensor that was saved in Python using
  //
  /*
python3 -c '
import torch
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
torch.save(t, "testdata/tensor_2x2.pt")'
  */
  std::vector<char> data;
  auto status = ReadFile("testdata/tensor_2x2.pt", data);
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_GT(data.size(), 0);
  auto val = torch::pickle_load(data);
  EXPECT_TRUE(val.isTensor());
  torch::Tensor t = val.toTensor();
  const auto dim = t.sizes();
  EXPECT_EQ(dim.size(), 2);
  EXPECT_EQ(dim[0], 2);
  EXPECT_EQ(dim[1], 2);
  torch::Tensor expected = torch::tensor({{1, 2}, {3, 4}});
  EXPECT_TRUE(torch::equal(t, expected));
}

TEST(MCTSTest, TorchPickleSave) {
  // Save a (2, 2) tensor into a protobuf bytes field.
  hexzpb::TrainingExample e;
  torch::Tensor t = torch::tensor({{1, 2}, {3, 4}});
  std::vector<char> data = torch::pickle_save(t);
  e.set_board(data.data(), data.size());
  EXPECT_GT(e.board().size(), 100);
}

}  // namespace
}  // namespace hexz
