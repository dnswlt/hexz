#include "mcts.h"

#include <absl/cleanup/cleanup.h>
#include <absl/status/status.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include "hexz.pb.h"

namespace hexz {

namespace {

using testing::ElementsAre;
using testing::ElementsAreArray;

class FakeModel : public Model {
 public:
  using Prediction = Model::Prediction;
  Prediction Predict(const Board& board, const Node& node) override {
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
  Node::uct_c = 1.0;
  Node n(nullptr, 0, Move{});
  std::vector<Move> moves{
      Move{0, 0, 0, 0},
      Move{0, 1, 0, 0},
  };
  n.CreateChildren(1 - n.turn(), moves);
  auto pr = torch::ones({2, 11, 10});
  pr = pr / pr.sum();
  n.SetMoveProbs(pr);
  auto& n_0 = *n.children()[0];
  auto& n_1 = *n.children()[1];
  // Mark victory for child n_0.
  n_0.Backpropagate(-1.0);
  Node* c = n.MaxPuctChild();
  ASSERT_TRUE(c != nullptr);
  //   EXPECT_EQ("FOO", n.DebugString());
  EXPECT_EQ(c, &n_0);
  EXPECT_EQ(c->parent(), &n);
  // PUCT should be greater than 1, the win rate.
  EXPECT_GT(c->Puct(), 1);
}

TEST(NodeTest, Backpropagate) {
  /*
  root (turn=0)
  - n_1 (turn=1)
    - n_1_1 (turn=0)
  - n_2 (player=1)

  Then backpropagate from n_1_1 and expect root to get updated.
  */
  Node root(nullptr, 0, Move{});
  std::vector<Move> root_moves{
      Move{0, 0, 0, 0},
      Move{1, 0, 0, 0},
  };
  root.CreateChildren(1 - root.turn(), root_moves);
  auto& n_1 = *root.children()[0];
  std::vector<Move> n_1_moves{
      Move{1, 1, 0, 0},
  };
  n_1.CreateChildren(1 - n_1.turn(), n_1_moves);
  auto& n_1_1 = *n_1.children()[0];
  // Player 0 won.
  n_1_1.Backpropagate(1.0);
  EXPECT_EQ(root.visit_count(), 1);
  EXPECT_EQ(n_1.visit_count(), 1);
  EXPECT_EQ(n_1_1.visit_count(), 1);
  EXPECT_EQ(root.wins(), 1.0);
  EXPECT_EQ(n_1.wins(), 0.0);
  EXPECT_EQ(n_1_1.wins(), 1.0);
}

TEST(NodeTest, FlatIndex) {
  // Tests that the flat indices point to the right array elements.
  // We test this by setting their move probabilities to increasing
  // values and asserting that the child nodes indeed have increasing priors.
  Node root(nullptr, 0, Move{});
  auto t = torch::arange(2 * 11 * 10).reshape({2, 11, 10});
  std::vector<Move> moves{
      Move{0, 0, 0, 0},
      Move{0, 0, 8, 0},
      Move{0, 2, 0, 0},
      Move{1, 0, 0, 0},
  };
  root.CreateChildren(1, moves);
  root.SetMoveProbs(t / t.sum());
  float prev = -1;
  for (const auto& c : root.children()) {
    ASSERT_LT(prev, c->Prior());
    prev = c->Prior();
  }
}

TEST(NodeTest, AddDirichletNoise) {
  Node root(nullptr, 0, Move{});
  std::vector<Move> moves{
      // Use the moves that come first in the flat representation of
      // move_probs_.
      Move{0, 0, 0, 0},
      Move{0, 0, 1, 0},
      Move{0, 0, 2, 0},
  };
  root.CreateChildren(1 - root.turn(), moves);
  auto t = torch::zeros({2, 11, 10});
  t.index_put_({0, 0, 0}, 30);
  t.index_put_({0, 0, 1}, 40);
  t.index_put_({0, 0, 2}, 10);
  t /= t.sum();
  root.SetMoveProbs(t);
  const std::vector<float> prior = root.move_probs();
  float prior_sum = std::accumulate(prior.begin(), prior.end(), 0.0);
  ASSERT_FLOAT_EQ(prior_sum, 1.0);
  root.AddDirichletNoise(0.5, 0.3);
  const std::vector<float> posterior = root.move_probs();
  // Expect that probs still sum to 1.
  float sum = std::accumulate(posterior.begin(), posterior.end(), 0.0);
  EXPECT_FLOAT_EQ(sum, 1.0);
  // Expect that values have changed.
  EXPECT_THAT(posterior, testing::Not(ElementsAreArray(prior)));
  // Only the elements representing child nodes should be modified:
  EXPECT_TRUE(std::all_of(posterior.begin() + 3, posterior.end(),
                          [](float x) { return x == 0.0; }));
  EXPECT_GT(posterior[0], 0);
  EXPECT_GT(posterior[1], 0);
  EXPECT_GT(posterior[2], 0);
}

TEST(NodeTest, ActionMask) {
  Node root(nullptr, 0, Move{});
  std::vector<Move> moves{
      // Can place a flag at (0, 0).
      Move{0, 0, 0, 0},
      // Can place a normal cell with value 1.0 at (7, 3).
      Move{1, 7, 3, 1.0},
  };
  root.CreateChildren(1 - root.turn(), moves);
  auto mask = root.ActionMask();
  ASSERT_THAT(mask.sizes(), ElementsAre(2, 11, 10));
  // Exactly two elements should be set:
  EXPECT_EQ(mask.flatten().nonzero().numel(), 2);
  EXPECT_TRUE(mask.index({0, 0, 0}).item<bool>());
  EXPECT_TRUE(mask.index({1, 7, 3}).item<bool>());
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
  EXPECT_EQ(mcts.NumRuns(0).first, 100);
  EXPECT_EQ(mcts.NumRuns(25).first, 75);
  EXPECT_EQ(mcts.NumRuns(50).first, 50);
}

TEST(MCTSTest, RemainingFlagsAreNotNegative) {
  // At some point the player for which moves were computed
  // and the player for which a move was made were mixed up.
  // This led to negative flag values. This test tries to capture
  // that the problem does not reoccur.
  Config config{
      .runs_per_move = 10,
      .runs_per_move_gradient = 0.0,
  };
  FakeModel fake_model;
  NeuralMCTS mcts(fake_model, config);
  Board b = Board::RandomBoard();
  auto root = std::make_unique<Node>(nullptr, 0, Move{-1, -1, -1, -1});
  for (int i = 0; i < 50; i++) {
    mcts.Run(*root, b, /*add_noise=*/false);
    ASSERT_GE(b.Flags(0), 0);
    ASSERT_GE(b.Flags(1), 0);
    int turn = root->turn();
    root = root->MostVisitedChildAsRoot();
    ASSERT_TRUE(b.Flags(turn) > 0 || root->move().typ != 0)
        << "Failed in move " << i;
    b.MakeMove(turn, root->move());
  }
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
      .runs_per_move = 20,
  };
  TorchModel model(scriptmodule);
  NeuralMCTS mcts(model, config);
  auto b = Board::RandomBoard();

  auto examples = mcts.PlayGame(b, /*max_runtime_seconds=*/0);
  ASSERT_TRUE(examples.ok());
  ASSERT_GE(examples->size(), 2);
  auto ex0 = (*examples)[0];
  auto ex1 = (*examples)[1];
  EXPECT_EQ(ex0.encoding(), hexzpb::TrainingExample::PYTORCH);
  EXPECT_GT(ex0.unix_micros(), 0);
  EXPECT_GT(ex0.stats().duration_micros(), 0);
  EXPECT_EQ(ex0.stats().visit_count(), config.runs_per_move);
  // Every game has 85 initial flag positions.
  EXPECT_EQ(ex0.stats().valid_moves(), 85);
  EXPECT_EQ(ex0.stats().move(), 0);
  EXPECT_EQ(ex1.stats().move(), 1);
  EXPECT_EQ(ex0.turn(), 0);
  EXPECT_EQ(ex1.turn(), 1);
  // Check board is a Tensor of the right shape.
  auto board_val = torch::pickle_load(
      std::vector<char>(ex0.board().begin(), ex0.board().end()));
  ASSERT_TRUE(board_val.isTensor());
  auto board = board_val.toTensor();
  EXPECT_EQ(board.sizes()[0], 11);
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

TEST(MCTSTest, WriteDotGraph) {
  // The file "testdata/scriptmodule.pt" is expected to be a ScriptModule of the
  // right shape to be used by NeuralMCTS.
  //
  // It can be generated with the regenerate.sh sidecar script.
  auto scriptmodule = torch::jit::load("testdata/scriptmodule.pt");
  scriptmodule.to(torch::kCPU);
  scriptmodule.eval();
  TorchModel model(scriptmodule);
  NeuralMCTS mcts(model, Config{});
  auto b = Board::RandomBoard();
  auto dot_path =
      std::filesystem::temp_directory_path() / "_MCTSTest_WriteDotGraph.dot";
  absl::Cleanup cleanup = [&dot_path]() { std::filesystem::remove(dot_path); };
  Node root(nullptr, 0, Move{});
  for (int i = 0; i < 100; i++) {
    mcts.RunReusingTree(root, b, /*add_noise=*/i == 0);
  }
  ASSERT_GT(root.children().size(), 0);
  auto status = WriteDotGraph(root, dot_path.string());
  ASSERT_TRUE(status.ok());
  // Expect that "something" was written.
  EXPECT_GT(std::filesystem::file_size(dot_path), 1000);
}

}  // namespace
}  // namespace hexz
