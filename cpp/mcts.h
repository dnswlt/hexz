#ifndef __HEXZ_MCTS_H__
#define __HEXZ_MCTS_H__
#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

#include "board.h"
#include "hexz.pb.h"
#include "util.h"

namespace hexz {

class Node {
 public:
  Node(Node* parent, int player, Move move);

  int Player() const { return player_; }
  const Move& GetMove() const { return move_; }

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
    move_probs_ =
        std::vector<float>(move_probs.data_ptr<float>(),
                           move_probs.data_ptr<float>() + move_probs.numel());
  }

 private:
  float Puct() const;

  Node* parent_;
  int player_;
  Move move_;
  int flat_idx_;
  float wins_;
  int visit_count_;
  std::vector<float> move_probs_;
  std::vector<std::unique_ptr<Node>> children_;
};

class NeuralMCTS {
  struct Prediction {
    torch::Tensor move_probs;
    float value;
  };

 public:
  NeuralMCTS(torch::jit::script::Module module);

  std::vector<hexzpb::TrainingExample> PlayGame(Board& board,
                                                int runs_per_move = 500,
                                                int max_moves = 200);

 private:
  bool Run(Node& root, Board& board);
  Prediction Predict(int player, const Board& board);

  torch::jit::script::Module module_;
};

}  // namespace hexz
#endif  // __HEXZ_MCTS_H__