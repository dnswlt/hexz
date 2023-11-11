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
  const std::vector<float>& move_probs() const noexcept { return move_probs_; }
  const Node* parent() const noexcept { return parent_; }
  float wins() const noexcept { return wins_; }
  const std::vector<std::unique_ptr<Node>>& children() const {
    return children_;
  }
  // Update the turn.
  void SetTurn(int turn) { turn_ = turn; }
  float Prior() const;

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
  void ShuffleChildren();

  // Returns the normalized visit counts (which sum to 1) as a (2, 11, 10)
  // tensor. This value should be used to update the move probs of the model.
  torch::Tensor NormVisitCounts() const;

  // Returns a boolean Tensor of shape (2, 11, 10) that is true in each
  // position that represents a valid move and false everywhere else.
  torch::Tensor ActionMask() const;

  // Sets the initial move probabilities ("policy") obtained from the model
  // prediction.
  void SetMoveProbs(torch::Tensor move_probs);
  // Adds Dirichlet noise to the (already set) move probs.
  // The weight parameter must be in the open interval (0, 1).
  // concentration is the Dirichlet concentration ("alpha") parameter.
  void AddDirichletNoise(float weight, float concentration);

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
  virtual Prediction Predict(const Board& board, const Node& node) = 0;
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
  Prediction Predict(const Board& board, const Node& node) override;

  const hexzpb::ModelKey& Key() const { return key_; }
  torch::jit::Module& Module() { return module_; }

 private:
  hexzpb::ModelKey key_;
  torch::jit::Module module_;
};

class NeuralMCTS {
 public:
  // C'tor parameters to construct a NeuralMCTS.
  struct Params {
    int runs_per_move = 800;
    double runs_per_move_gradient = 0.0;
    int max_moves_per_game = 200;
    // Dirichlet noise is only added if this value is >0.
    float dirichlet_concentration = 0.0;
  };
  // The model is not owned. Owners of the NeuralMCTS instance must ensure it
  // outlives this instance.
  NeuralMCTS(Model& model, const Params& params);

  absl::StatusOr<std::vector<hexzpb::TrainingExample>> PlayGame(
      Board& board, int max_runtime_seconds);

  // Returns the number of times Run should be called for the given move number.
  // This is only relevant for the "decay" of Run calls in a full PlayGame
  // cycle.
  int NumRuns(int move) const noexcept;

  // Executes a single run of the MCTS algorithm, starting at root.
  // If add_noise is true, Dirichlet noise will be added to the root node's
  // move probs.
  bool Run(Node& root, const Board& board, bool add_noise);

  // SuggestMove returns the best move suggestion that the NeuralMCTS algorithm
  // comes up with in think_time_millis milliseconds.
  absl::StatusOr<std::unique_ptr<Node>> SuggestMove(int player,
                                                    const Board& board,
                                                    int think_time_millis);

 private:
  int runs_per_move_;
  double runs_per_move_gradient_;
  int max_moves_per_game_;
  float dirichlet_concentration_;
  // Not owned.
  Model& model_;
};

}  // namespace hexz
#endif  // __HEXZ_MCTS_H__