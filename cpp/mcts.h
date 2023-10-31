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
  void SetMoveProbs(torch::Tensor move_probs) { move_probs_ = move_probs; }

 private:
  float Puct() const;

  Node* parent_;
  int player_;
  Move move_;
  float wins_;
  int visit_count_;
  torch::Tensor move_probs_;
  std::vector<std::unique_ptr<Node>> children_;
};

class NeuralMCTS {
  struct Prediction {
    torch::Tensor move_probs;
    float value;
  };
  struct PerfStats {
    std::string label;
    int64_t acc_duration_micros = 0;
    int64_t count = 0;
    void Record(int64_t start_micros) {
        acc_duration_micros += UnixMicros() - start_micros;
        count++;
    }
  };

 public:
  NeuralMCTS(torch::jit::script::Module module);

  Prediction Predict(int player, const Board& board);
  bool Run(Node* root, const Board& board);
  std::vector<hexzpb::TrainingExample> PlayGame(const Board& board,
                                                int runs_per_move = 500,
                                                int max_moves = 200);
  std::vector<PerfStats> GetPerfStats() const;

 private:
  enum PerfStatsIdx {
    PS_Predict = 0,
    PS_FindLeaf = 1,
    PS_MakeMove = 2,
    PerStatsSize = 3,
  };

  torch::jit::script::Module module_;
  PerfStats pstats_[PerStatsSize];
};

}  // namespace hexz
#endif  // __HEXZ_MCTS_H__