
#include "mcts.h"

#include <absl/cleanup/cleanup.h>
#include <absl/status/status.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <type_traits>
#include <vector>

#include "base.h"
#include "hexz.pb.h"
#include "model.h"

namespace hexz {

namespace {

namespace fs = std::filesystem;

using testing::ElementsAre;
using testing::ElementsAreArray;

const hexzpb::ModelKey& TestModelKey() {
  static hexzpb::ModelKey* key = [] {
    auto* k = new hexzpb::ModelKey();
    k->set_name("test_model");
    k->set_checkpoint(0);
    return k;
  }();
  return *key;
}

class FakeModel : public Model {
  void UpdateModel(hexzpb::ModelKey key, torch::jit::Module&& module) override {
  }
  hexzpb::ModelKey Key() const override { return TestModelKey(); }
};

class UniformFakeModel final : public FakeModel {
 public:
  // Always returns uniform probabilities.
  Prediction Predict(torch::Tensor board, torch::Tensor action_mask) override {
    auto t = torch::ones({2, 11, 10}).where(action_mask, 0);
    t = t / t.sum();
    return Prediction{
        .move_probs = t,
        .value = 0.0,
    };
  }
};

class ConstantFakeModel final : public FakeModel {
 public:
  ConstantFakeModel(torch::Tensor move_probs, float value)
      : move_probs_{move_probs}, value_{value} {
    auto dim = move_probs.sizes();
    ABSL_CHECK(dim.size() == 3);
    ABSL_CHECK(dim[0] == 2 && dim[1] == 11 && dim[2] == 10);
  }
  Prediction Predict(torch::Tensor board, torch::Tensor action_mask) override {
    auto t = move_probs_.where(action_mask, 0);
    t /= t.sum();
    return Prediction{
        .move_probs = t,
        .value = value_,
    };
  }

 private:
  torch::Tensor move_probs_;
  float value_;
};

// A PlayoutRunner useful for tests. You can specify the results
// it is supposed to return.
// The runner will return the same specified result for each call to Run,
// and advance its position in the results by one on each call to ResetStats.
// This is an unfortunate implementation detail leak - we know that
// NeuralMCTS::PlayGame call ResetStats on each move.
class FakePlayoutRunner final : public PlayoutRunner {
 public:
  explicit FakePlayoutRunner(std::vector<float> results) : results_{results} {}
  Stats Run(const Board& board, int turn, int runs) override {
    ABSL_CHECK(iteration_ >= 0 && iteration_ < results_.size());
    float r = results_[iteration_];
    Stats s{
        .result_sum = runs * r,
        .runs = runs,
    };
    stats_.Merge(s);
    return s;
  }
  const Stats& AggregatedStats() const override { return stats_; }
  void ResetStats() override {
    iteration_++;
    stats_ = Stats{};
  }

 private:
  Stats stats_;
  std::vector<float> results_;
  int iteration_ = -1;
};

TEST(NodeTest, InitializedToZero) {
  Node n(nullptr, 0, Move{});
  EXPECT_EQ(n.visit_count(), 0);
}

TEST(NodeTest, IsLeaf) {
  Node n(nullptr, 0, Move{});
  EXPECT_TRUE(n.IsLeaf());
  std::vector<Move> moves{
      Move::Flag(0, 0),
  };
  n.CreateChildren(moves);
  EXPECT_FALSE(n.IsLeaf());
}

TEST(NodeTest, MaxPuctChild) {
  Node::uct_c = 1.0;
  Node::initial_root_q_value = -0.2;
  Node::initial_q_penalty = 0.3;
  Node n(0);
  std::vector<Move> moves{
      Move::Flag(0, 0),
      Move::Flag(1, 0),
  };
  n.CreateChildren(moves);
  auto pr = torch::ones({2, 11, 10});
  pr = pr / pr.sum();
  n.SetMoveProbs(pr);
  auto& n_0 = *n.children()[0];
  auto& n_1 = *n.children()[1];
  // Mark victory for child n_0.
  n_0.Backpropagate(1.0);
  Node* c = n.MaxPuctChild();
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ(c, &n_0);
  EXPECT_EQ(c->parent(), &n);
  // PUCT should be greater than 1, the win rate.
  EXPECT_GT(c->Puct(), 1);
}

TEST(NodeTest, MaxPuctChildDepth2) {
  Node::uct_c = 1.0;
  Node::initial_root_q_value = -0.2;
  Node::initial_q_penalty = 0.3;
  Node root(0);
  std::vector<Move> root_moves{
      Move::Flag(0, 0),
      Move::Flag(1, 0),
  };
  root.CreateChildren(root_moves);
  auto pr = torch::ones({2, 11, 10});
  pr = pr / pr.sum();
  root.SetMoveProbs(pr);
  auto& n_0 = *root.children()[0];
  auto& n_1 = *root.children()[1];
  std::vector<Move> n_0_moves{
      Move{Move::Typ::kNormal, 2, 0, 0},
      Move{Move::Typ::kNormal, 3, 0, 0},
  };
  n_0.CreateChildren(n_0_moves);
  auto& n_0_0 = *n_0.children()[0];
  n_0_0.Backpropagate(-1.0);  // P1 wins.

  Node* c = root.MaxPuctChild();
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ(c, &n_1);
  // Should be the initial Q value + the prior, since uct_c and the visit count
  // factor should be 1.
  EXPECT_FLOAT_EQ(c->Puct(), Node::initial_root_q_value +
                                 pr.index({0, 1, 0}).item<float>());
}

TEST(NodeTest, Backpropagate) {
  /*
  root (turn=0)
  - n_1 (turn=1)
    - n_1_1 (turn=0)
  - n_2 (player=1)

  Then backpropagate from n_1_1 and expect root to get updated.
  */
  Node root(0);
  std::vector<Move> root_moves{
      Move::Flag(0, 0),
      Move{Move::Typ::kNormal, 0, 0, 0},
  };
  root.CreateChildren(root_moves);
  auto& n_1 = *root.children()[0];
  std::vector<Move> n_1_moves{
      Move{Move::Typ::kNormal, 1, 0, 0},
  };
  n_1.CreateChildren(n_1_moves);
  auto& n_1_1 = *n_1.children()[0];
  // Player 0 won.
  n_1_1.Backpropagate(1.0);
  EXPECT_EQ(root.visit_count(), 1);
  EXPECT_EQ(n_1.visit_count(), 1);
  EXPECT_EQ(n_1_1.visit_count(), 1);
  EXPECT_EQ(root.wins(), 0.0)
      << "Root wins should not get updated, they are meaningless";
  EXPECT_EQ(n_1.wins(), 1.0);
  EXPECT_EQ(n_1_1.wins(), -1.0);
}

TEST(NodeTest, BackpropagateFraction) {
  /*
  Like Backpropagate, but propagates back only a fractional value,
  as is common when backpropagating model predictions.

  root (turn=0)
  - n_1 (turn=1)
    - n_1_1 (turn=0)
  - n_2 (player=1)

  Then backpropagate from n_1_1.
  */
  Node root(0);
  std::vector<Move> root_moves{
      Move::Flag(0, 0),
      Move{Move::Typ::kNormal, 0, 0, 0},
  };
  root.CreateChildren(root_moves);
  auto& n_1 = *root.children()[0];
  std::vector<Move> n_1_moves{
      Move{Move::Typ::kNormal, 1, 0, 0},
  };
  n_1.CreateChildren(n_1_moves);
  auto& n_1_1 = *n_1.children()[0];
  // Player 0 won.
  n_1_1.Backpropagate(0.2);
  EXPECT_EQ(root.visit_count(), 1);
  EXPECT_EQ(n_1.visit_count(), 1);
  EXPECT_EQ(n_1_1.visit_count(), 1);
  EXPECT_FLOAT_EQ(root.wins(), 0.0);
  EXPECT_FLOAT_EQ(n_1.wins(), 0.2);
  EXPECT_FLOAT_EQ(n_1_1.wins(), -0.2);
}

TEST(NodeTest, FlatIndex) {
  // Tests that the flat indices point to the right array elements.
  // We test this by setting their move probabilities to increasing
  // values and asserting that the child nodes indeed have increasing priors.
  Node root(nullptr, 0, Move{});
  auto t = torch::arange(2 * 11 * 10).reshape({2, 11, 10});
  std::vector<Move> moves{
      Move::Flag(0, 0),
      Move::Flag(0, 8),
      Move::Flag(2, 0),
      Move{Move::Typ::kNormal, 0, 0, 0},
  };
  root.CreateChildren(moves);
  root.SetMoveProbs(t / t.sum());
  float prev = -1;
  for (const auto& c : root.children()) {
    ASSERT_LT(prev, c->prior());
    prev = c->prior();
  }
}

TEST(NodeTest, AddDirichletNoise) {
  internal::RNG rng;
  Node root(nullptr, 0, Move{});
  std::vector<Move> moves{
      // Use the moves that come first in the flat representation of
      // move_probs_.
      Move::Flag(0, 0),
      Move::Flag(0, 1),
      Move::Flag(0, 2),
  };
  root.CreateChildren(moves);
  auto t = torch::zeros({2, 11, 10});
  t.index_put_({0, 0, 0}, 30);
  t.index_put_({0, 0, 1}, 40);
  t.index_put_({0, 0, 2}, 10);
  t /= t.sum();
  root.SetMoveProbs(t);
  std::vector<float> prior;
  for (const auto& c : root.children()) {
    prior.push_back(c->prior());
  }
  float prior_sum = std::accumulate(prior.begin(), prior.end(), 0.0);
  ASSERT_FLOAT_EQ(prior_sum, 1.0);
  root.AddDirichletNoise(0.5, 0.3, rng);
  std::vector<float> posterior;
  for (const auto& c : root.children()) {
    posterior.push_back(c->prior());
  }
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
      Move::Flag(0, 0),
      // Can place a normal cell with value 1.0 at (7, 3).
      Move{Move::Typ::kNormal, 7, 3, 1.0},
  };
  root.CreateChildren(moves);
  auto mask = root.ActionMask();
  ASSERT_THAT(mask.sizes(), ElementsAre(2, 11, 10));
  // Exactly two elements should be set:
  EXPECT_EQ(mask.flatten().nonzero().numel(), 2);
  EXPECT_TRUE(mask.index({0, 0, 0}).item<bool>());
  EXPECT_TRUE(mask.index({1, 7, 3}).item<bool>());
}

class MCTSScriptModuleTest : public testing::Test {
 protected:
  void SetUp() override {
    // The file "testdata/scriptmodule.pt" is expected to be a ScriptModule of
    // the right shape to be used by NeuralMCTS.
    //
    // It can be generated with the regenerate.sh sidecar script.
    scriptmodule_ = torch::jit::load("testdata/scriptmodule.pt");
    scriptmodule_.to(torch::kCPU);
    scriptmodule_.eval();
  }

  torch::jit::Module scriptmodule_;
};

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
  // Normal config should return the configured runs.
  Config config{
      .runs_per_move = 100,
  };
  UniformFakeModel fake_model;
  NeuralMCTS mcts(fake_model, /*playout_runner=*/nullptr, config);
  EXPECT_EQ(mcts.NumRuns(0).first, 100);
  EXPECT_EQ(mcts.NumRuns(25).first, 100);
  // Fast run config should return # of fast runs, but only after the
  config = Config{
      .runs_per_move = 100,
      .runs_per_fast_move = 10,
      .fast_move_prob = 1.0,  // only fast moves
  };
  NeuralMCTS mcts_fast(fake_model, /*playout_runner=*/nullptr, config);
  EXPECT_EQ(mcts_fast.NumRuns(0).first, 100);
  EXPECT_EQ(mcts_fast.NumRuns(kFirstFastMove).first, 10);
}

TEST(MCTSTest, RemainingFlagsAreNotNegative) {
  // At some point the player for which moves were computed
  // and the player for which a move was made were mixed up.
  // This led to negative flag values. This test tries to capture
  // that the problem does not reoccur.
  Config config{
      .runs_per_move = 10,
  };
  internal::RNG rng;
  UniformFakeModel fake_model;
  NeuralMCTS mcts(fake_model, /*playout_runner=*/nullptr, config);
  Board b = Board::RandomBoard();
  auto root = std::make_unique<Node>(nullptr, 0, Move{});
  for (int i = 0; i < 50; i++) {
    mcts.Run(*root, b);
    ASSERT_GE(b.Flags(0), 0);
    ASSERT_GE(b.Flags(1), 0);
    int turn = root->NextTurn();
    root = root->SelectChildAsRoot(root->SelectChildForNextMove(rng));
    ASSERT_TRUE(b.Flags(turn) > 0 || root->move().typ != Move::Typ::kFlag)
        << "Failed in move " << i;
    b.MakeMove(turn, root->move());
  }
}

TEST_F(MCTSScriptModuleTest, PlayGame) {
  Perfm::InitScope perfm;
  Config config{
      .runs_per_move = 50,
      .dirichlet_concentration = 0.3,
      .random_playouts = 10,
      .resign_threshold = std::numeric_limits<float>::max(),  // never resign
  };
  TorchModel model(std::move(scriptmodule_));
  NeuralMCTS mcts(model, std::make_unique<RandomPlayoutRunner>(), config);
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
  EXPECT_EQ(ex0.move().move(), 0);
  EXPECT_EQ(ex1.move().move(), 1);
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

TEST_F(MCTSScriptModuleTest, PlayGameFiberTorchModel) {
  // Smoke test to ensure NeuralMCTS works with the FiberTorchModel.
  Config config{
      .runs_per_move = 50,
      .dirichlet_concentration = 0.3,
      .random_playouts = 10,
      .resign_threshold = std::numeric_limits<float>::max(),  // never resign
  };
  hexzpb::ModelKey key;
  FiberTorchModel model(key, std::move(scriptmodule_), torch::kCPU, 1);
  auto token = model.RegisterThread();
  NeuralMCTS mcts(model, std::make_unique<RandomPlayoutRunner>(), config);
  auto b = Board::RandomBoard();

  auto examples = mcts.PlayGame(b, /*max_runtime_seconds=*/0);
  EXPECT_TRUE(examples.ok());
  EXPECT_GE(examples->size(), 2);
}

TEST(MCTSTest, PlayGameStats) {
  // Test that TrainingExample stats are populated as expected.
  Config config{
      .runs_per_move = 50,
      .dirichlet_concentration = 0.0,
      .random_playouts = 0,
      .resign_threshold = std::numeric_limits<float>::max(),  // never resign
  };
  auto move_probs = torch::ones({2, 11, 10});
  move_probs /= move_probs.sum();
  float value = 1.0;  // P0 always wins.
  ConstantFakeModel fake_model(move_probs, value);
  NeuralMCTS mcts(fake_model, std::make_unique<RandomPlayoutRunner>(), config);
  auto b = Board::RandomBoard();

  auto examples = mcts.PlayGame(b, /*max_runtime_seconds=*/0);
  ASSERT_TRUE(examples.ok());
  ASSERT_GE(examples->size(), 1);
  auto ex = (*examples)[0];
  ASSERT_TRUE(ex.has_stats());
  const auto& stats = ex.stats();
  EXPECT_EQ(stats.visit_count(), config.runs_per_move);
  // Every game has 85 initial flag positions.
  EXPECT_EQ(stats.valid_moves(), 85);
  // At least one child node should have been visited.
  EXPECT_GT(stats.visited_children(), 0);
  EXPECT_GT(stats.max_child_vc(), 0);
  EXPECT_NE(stats.selected_child_q(), 0.0);
  EXPECT_GT(stats.selected_child_vc(), 0);
  // At least nodes up to level 3 should have been visited
  EXPECT_GE(stats.nodes_per_depth().size(), 3);
}

TEST(MCTSTest, PlayGameResign) {
  // If all random playouts indicate the same winner, the game should be
  // resigned.
  Config config{
      .runs_per_move = 50,
      .fast_move_prob = 0.0,  // but no fast moves (to avoid interference)
      // Avoid penalties in this test: with fake models returning uniform values
      // for all nodes, the search would otherwise just go down a single path.
      .initial_root_q_value = 0.0,
      .initial_q_penalty = 0.0,
      .dirichlet_concentration = 0.0,
      .random_playouts = 10,  // use random playouts
      .resign_threshold = 0.999,
  };
  UniformFakeModel model;
  // Playouts yield a draw on move 0, a very weak value on move 1,
  // then player 1 wins every game in move 2, which should lead to resignation.
  auto runner =
      std::make_unique<FakePlayoutRunner>(std::vector<float>{0.0, -0.01, -1});
  NeuralMCTS mcts(model, std::move(runner), config);
  auto b = Board::RandomBoard();
  auto examples = mcts.PlayGame(b, /*max_runtime_seconds=*/0);
  ASSERT_TRUE(examples.ok());
  EXPECT_EQ(examples->size(), 2) << "Should have resigned on the third move";
}

TEST_F(MCTSScriptModuleTest, WriteDotGraph) {
  TorchModel model(std::move(scriptmodule_));
  NeuralMCTS mcts(model, /*playout_runner=*/nullptr, Config{});
  auto b = Board::RandomBoard();
  auto dot_path =
      fs::temp_directory_path() / fs::path("_MCTSTest_WriteDotGraph.dot");
  absl::Cleanup cleanup = [&dot_path]() { fs::remove(dot_path); };
  Node root(nullptr, 0, Move{});
  for (int i = 0; i < 100; i++) {
    mcts.SelfplayRun(root, b, /*add_noise=*/i == 0, /*run_playouts=*/false);
  }
  ASSERT_GT(root.children().size(), 0);
  auto status = WriteDotGraph(root, dot_path.string());
  ASSERT_TRUE(status.ok());
  // Expect that "something" was written.
  EXPECT_GT(fs::file_size(dot_path), 1000);
}

}  // namespace
}  // namespace hexz
