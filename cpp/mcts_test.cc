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

class FakeModel : public Model {
 public:
  using Prediction = Model::Prediction;
  Prediction Predict(int player, const Board& board) override {
    auto t = torch::ones({2, 11, 10});
    t = t / t.sum();
    return Prediction{
        .move_probs = t,
        .value = 0.0,
    };
  }
};

TEST(NodeTest, InitializedToZero) {
  Node n(nullptr, 0, Move{});
  EXPECT_EQ(n.visit_count(), 0);
}

TEST(NodeTest, IsLeaf) {
  Node n(nullptr, 0, Move{});
  EXPECT_TRUE(n.IsLeaf());
  std::vector<Move> moves{
      Move{0, 0, 0, 0},
  };
  n.CreateChildren(0, moves);
  EXPECT_FALSE(n.IsLeaf());
}

TEST(NodeTest, MaxPuctChild) {
  Node n(nullptr, 0, Move{});
  std::vector<Move> moves{
      Move{0, 0, 0, 0},
  };
  n.CreateChildren(0, moves);
  auto pr = torch::ones({2, 11, 10});
  pr = pr / pr.sum();
  n.SetMoveProbs(pr);
  Node* c = n.MaxPuctChild();
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ(c->parent(), &n);
  EXPECT_EQ(c->Puct(), 0);
}

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
  // Load a (2, 2) Tensor that was saved in Python.
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

TEST(MCTSTest, NumRuns) {
  Config config{
      .runs_per_move = 100,
      .runs_per_move_gradient = -0.01,
  };
  FakeModel fake_model;
  NeuralMCTS mcts(fake_model, config);
  EXPECT_EQ(mcts.NumRuns(0), 100);
  EXPECT_EQ(mcts.NumRuns(25), 75);
  EXPECT_EQ(mcts.NumRuns(50), 50);
}

TEST(MCTSTest, PlayGame) {
  // The file "testdata/scriptmodule.pt" is expected to be a ScriptModule of the
  // right shape to be used by NeuralMCTS.
  //
  // It can be generated with the regenerate.sh sidecar script.
  auto scriptmodule = torch::jit::load("testdata/scriptmodule.pt");
  scriptmodule.to(torch::kCPU);
  scriptmodule.eval();
  Config config{
      .runs_per_move = 10,
  };
  TorchModel model(scriptmodule);
  NeuralMCTS mcts(model, config);
  auto b = Board::RandomBoard();

  auto examples = mcts.PlayGame(b, /*max_runtime_seconds=*/0);
  ASSERT_TRUE(examples.ok());
  ASSERT_GT(examples->size(), 0);
  auto ex0 = (*examples)[0];
  EXPECT_EQ(ex0.encoding(), hexzpb::TrainingExample::PYTORCH);
  EXPECT_GT(ex0.unix_micros(), 0);
  EXPECT_GT(ex0.stats().duration_micros(), 0);
  EXPECT_EQ(ex0.stats().visit_count(), 10);
  EXPECT_EQ(ex0.stats().valid_moves(), 85);  // Every game has 85 initial flag positions.
  EXPECT_EQ(ex0.stats().move(), 0);
  // Check board is a Tensor of the right shape.
  auto board_val = torch::pickle_load(
      std::vector<char>(ex0.board().begin(), ex0.board().end()));
  ASSERT_TRUE(board_val.isTensor());
  auto board = board_val.toTensor();
  EXPECT_EQ(board.sizes()[0], 9);
  EXPECT_EQ(board.sizes()[1], 11);
  EXPECT_EQ(board.sizes()[2], 10);
  // Check move_probs is a Tensor of the right shape.
  auto pr_val = torch::pickle_load(
      std::vector<char>(ex0.move_probs().begin(), ex0.move_probs().end()));
  ASSERT_TRUE(pr_val.isTensor());
  auto pr = pr_val.toTensor();
  EXPECT_EQ(pr.sizes()[0], 2);
  EXPECT_EQ(pr.sizes()[1], 11);
  EXPECT_EQ(pr.sizes()[2], 10);
}

}  // namespace
}  // namespace hexz
