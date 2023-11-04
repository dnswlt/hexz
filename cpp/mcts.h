#ifndef __HEXZ_MCTS_H__
#define __HEXZ_MCTS_H__

#include <absl/status/statusor.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

#include "base.h"
#include "board.h"
#include "hexz.pb.h"
#include "perfm.h"

namespace hexz {

class Node {
 public:
  Node(Node* parent, int player, Move move);

  int Player() const { return player_; }
  const Move& GetMove() const { return move_; }

  float Puct() const noexcept;
  Node* MaxPuctChild();
  // Returns a pointer to the child with the greated visit count,
  // marks it as a root node (by setting its parent_ to nullptr),
  // and clears the vector of child nodes of *this* Node.
  //
  // The Node on which this method was called MUST NOT be used
  // afterwards!
  std::unique_ptr<Node> MostVisitedChildAsRoot();

  void Backpropagate(float result);

  bool IsLeaf() const { return children_.empty(); }

  // Adds children representing the given moves to this node.
  // Children will be shuffled randomly, to avoid selection bias.
  void CreateChildren(int player, const std::vector<Move>& moves);

  torch::Tensor MoveProbs() {
    assert(move_probs_.size() == 2 * 11 * 10);
    return torch::from_blob(move_probs_.data(), {2, 11, 10});
  }
  void SetMoveProbs(torch::Tensor move_probs) {
    assert(move_probs.numel() == 2 * 11 * 10);
    assert(std::abs(move_probs.sum().item<float>() - 1.0) < 1e-3);
    move_probs_ =
        std::vector<float>(move_probs.data_ptr<float>(),
                           move_probs.data_ptr<float>() + move_probs.numel());
  }

  int visit_count() const noexcept { return visit_count_; }
  const Node* parent() const noexcept { return parent_; }
  // Returns the number of children that had a nonzero visit_count.
  int NumVisitedChildren() const noexcept;
  std::string Stats() const;
  // Weight of the exploration term.
  // Must only be modified at program startup.
  static float uct_c;

 private:
  Node* parent_;
  int player_;
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
      : key_{key}, module_{module}  {}
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

 private:
  bool Run(Node& root, Board& board);

  int runs_per_move_;
  double runs_per_move_gradient_;
  int max_moves_per_game_;

  // Not owned.
  Model& model_;
};

}  // namespace hexz
#endif  // __HEXZ_MCTS_H__