#ifndef __HEXZ_MCTS_H__
#define __HEXZ_MCTS_H__

#include <absl/status/statusor.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cassert>
#include <ostream>
#include <vector>

#include "base.h"
#include "board.h"
#include "hexz.pb.h"
#include "perfm.h"

namespace hexz {

class Node {
 public:
  Node(Node* parent, int turn, Move move);

  // Simple getters.
  int turn() const { return turn_; }
  const Move& move() const { return move_; }
  int visit_count() const noexcept { return visit_count_; }
  const Node* parent() const noexcept { return parent_; }
  float wins() const noexcept { return wins_; }
  std::vector<std::unique_ptr<Node>>& children() { return children_; }
  // Update the turn.
  void SetTurn(int turn) { turn_ = turn; }

  float Puct() const noexcept;
  Node* MaxPuctChild() const;
  // Returns a pointer to the child with the greated visit count,
  // marks it as a root node (by setting its parent_ to nullptr),
  // and clears the vector of child nodes of *this* Node.
  //
  // The Node on which this method was called MUST NOT be used
  // afterwards!
  std::unique_ptr<Node> MostVisitedChildAsRoot();

  // Backpropagate propagates the given result to this Node and all parents.
  // The given result is interpreted from the perspective of player 0.
  // A value of 1 always means that player 0 won.
  // This is particularly relevant when feeding in model predictions,
  // which are usually from the perspective of the current player.
  void Backpropagate(float result);

  bool IsLeaf() const { return children_.empty(); }

  // Adds children representing the given moves to this node.
  void CreateChildren(int turn, const std::vector<Move>& moves);
  // Shuffles children randomly. This can be used to avoid selection bias.
  void ShuffleChildren() {
    std::shuffle(children_.begin(), children_.end(), internal::rng);
  }

  // Returns the normalized visit counts (which sum to 1) as a (2, 11, 10)
  // tensor. This value should be used to update the move probs of the model.
  torch::Tensor NormVisitCounts() {
    auto t = torch::zeros({2, 11, 10});
    auto t_acc = t.accessor<float, 3>();
    for (const auto& c : children_) {
      const auto& m = c->move_;
      t_acc[m.typ][m.r][m.c] = static_cast<float>(c->visit_count_);
    }
    t /= t.sum();
    return t;
  }

  // Sets the initial move probabilities ("policy") obtained from the model
  // prediction.
  void SetMoveProbs(torch::Tensor move_probs) {
    assert(move_probs.numel() == 2 * 11 * 10);
    assert(std::abs(move_probs.sum().item<float>() - 1.0) < 1e-3);
    move_probs_ =
        std::vector<float>(move_probs.data_ptr<float>(),
                           move_probs.data_ptr<float>() + move_probs.numel());
  }

  // Returns the number of children that had a nonzero visit_count.
  int NumVisitedChildren() const noexcept;
  int NumChildren() const noexcept { return children_.size(); }
  std::string Stats() const;
  // Weight of the exploration term.
  // Must only be modified at program startup.
  static float uct_c;

  std::string DebugString() const;

 private:
  void AppendDebugString(std::ostream& os, const std::string& indent) const;

  Node* parent_;
  int turn_;
  Move move_;
  int flat_idx_;
  float wins_ = 0.0;
  int visit_count_ = 0;
  std::vector<float> move_probs_;
  std::vector<std::unique_ptr<Node>> children_;
};

// Interface class for a model than can make predictions for
// move likelihoods and board evaluations.
class Model {
 public:
  struct Prediction {
    torch::Tensor move_probs;
    float value;
  };
  virtual Prediction Predict(int player, const Board& board) = 0;
  virtual ~Model() = default;
};

// Implementation of Model that uses an actual PyTorch ScriptModule.
// This is the implementation that should be used everywhere outside of tests.
class TorchModel : public Model {
  using Prediction = Model::Prediction;

 public:
  explicit TorchModel(torch::jit::Module module) : module_{module} {}
  TorchModel(hexzpb::ModelKey key, torch::jit::Module module)
      : key_{key}, module_{module} {}
  Prediction Predict(int player, const Board& board) override;

  const hexzpb::ModelKey& Key() const { return key_; }
  torch::jit::Module& Module() { return module_; }

 private:
  hexzpb::ModelKey key_;
  torch::jit::Module module_;
};

class NeuralMCTS {
 public:
  // The model is not owned. Owners of the NeuralMCTS instance must ensure it
  // outlives this instance.
  NeuralMCTS(Model& model, const Config& config);

  absl::StatusOr<std::vector<hexzpb::TrainingExample>> PlayGame(
      Board& board, int max_runtime_seconds);

  int NumRuns(int move) const noexcept;
  bool Run(Node& root, Board& board);

 private:
  int runs_per_move_;
  double runs_per_move_gradient_;
  int max_moves_per_game_;

  // Not owned.
  Model& model_;
};

}  // namespace hexz
#endif  // __HEXZ_MCTS_H__