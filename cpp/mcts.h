#pragma once

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include <cassert>
#include <ostream>
#include <queue>
#include <vector>

#include "base.h"
#include "batch.h"
#include "board.h"
#include "hexz.pb.h"
#include "model.h"
#include "perfm.h"

namespace hexz {

// Some constants for which I've seen no need to tweak them dynamically.
// Most of these either get deleted or end up as Config attributes :)
constexpr float kNoiseWeight = 0.25;
// The first move for which a fast move is possible.
constexpr int kFirstFastMove = 6;

// All code that must be run once at program startup should
// be added to this function.
void InitializeFromConfig(const Config& config);

class Node {
 public:
  explicit Node(int turn) : Node(nullptr, turn, Move{}) {}
  Node(Node* parent, int turn, Move move);

  // Must be called at startup to initialize static configuration parameters.
  static void InitializeStaticMembers(const Config& config) {
    uct_c = config.uct_c;
    initial_root_q_value = config.initial_root_q_value;
    initial_q_penalty = config.initial_q_penalty;
  }

  // Simple getters.
  const Move& move() const { return move_; }
  float prior() const noexcept { return prior_; }
  int visit_count() const noexcept { return visit_count_; }
  const Node* parent() const noexcept { return parent_; }
  float wins() const noexcept { return wins_; }
  float value() const noexcept { return value_; }
  float ValueP0() const noexcept { return turn_ == 0 ? value_ : -value_; }
  bool terminal() const noexcept { return terminal_; }
  const std::vector<std::unique_ptr<Node>>& children() const {
    return children_;
  }

  // Returns this node's Q value. If the child has not been visited
  // yet, returns a Q value derived from the parent's Q value.
  float Q() const noexcept;

  // Returns the player whose turn it is to make the move stored
  // in this node. MUST NOT be called on the root node.
  int MoveTurn() const noexcept;
  // Returns the player whose turn it is after the move stored
  // in this node was made. This may be wrong ONLY in leaf nodes, b/c
  // Flagz isn't strictly alternating (a player can run out of moves).
  int NextTurn() const { return turn_; }
  // Updates the turn.
  void SetNextTurn(int turn) { turn_ = turn; }

  // Returns this node's PUCT value.
  float Puct() const noexcept;
  // Returns a non-owned pointer to the child with the greatest PUCT value.
  Node* MaxPuctChild() const;

  // Selects a child node with a probability
  // proportional to its relative visit count.
  int SampleChild() const;

  // Returns the specified child and marks it as a root node (by
  // setting its parent_ to nullptr), and moves the child node out of the
  // children vector of *this* Node.
  //
  // The Node on which this method was called MUST NOT be used
  // afterwards!
  std::unique_ptr<Node> SelectChildAsRoot(int i);

  // Backpropagate propagates the given result to this Node and all parents.
  // The given result is interpreted from the perspective of player 0.
  // A value of 1 always means that player 0 won.
  // This is particularly relevant when feeding in model predictions,
  // which are usually from the perspective of the current player.
  void Backpropagate(float result);

  bool IsRoot() const noexcept { return parent_ == nullptr; }
  bool IsLeaf() const noexcept { return children_.empty(); }

  // Adds children representing the given moves to this node.
  void CreateChildren(const std::vector<Move>& moves);
  // Shuffles children randomly. This can be used to avoid selection bias.
  void ShuffleChildren();

  // Returns the normalized visit counts (which sum to 1) as a (2, 11, 10)
  // tensor. This value should be used to update the move probs of the model.
  torch::Tensor NormVisitCounts() const;

  // Recursively zeros visit_count_ and wins_ of all nodes in the whole subtree.
  // This method is used for prediction caching: re-use the search tree of
  // the previous move during self-play, but only for re-using the model
  // predictions.
  void ResetTree();

  // Returns a boolean Tensor of shape (2, 11, 10) that is true in each
  // position that represents a valid move and false everywhere else.
  torch::Tensor ActionMask() const;

  // Sets the initial move probabilities ("policy") obtained from the model
  // prediction.
  void SetMoveProbs(torch::Tensor move_probs);
  // Returns the prior move probabilities as a (2, 11, 10) tensor.
  // The priors will include Dirichlet noise, if it was added.
  torch::Tensor Priors() const;

  // Sets the value of this node (typically: as predicted by the model).
  // The value should range between -1 and 1; values greater than 0 predict that
  // it is a win for the player whose turn it is in this node.
  void SetValue(float value) { value_ = value; }
  void SetTerminal(bool b) { terminal_ = b; }

  // Adds Dirichlet noise to the (already set) move probs.
  // The weight parameter must be in the open interval (0, 1).
  // concentration is the Dirichlet concentration ("alpha") parameter.
  void AddDirichletNoise(float weight, float concentration);

  // Returns the number of children that had a nonzero visit_count.
  int NumVisitedChildren() const noexcept;
  void PopulateStats(hexzpb::TrainingExample::Stats& stats) const;
  std::string Stats() const;

  std::string DebugString() const;

  // Static members that server as "configurable constants". They all must
  // only be modified at program startup.
  //
  // See the corresponding field in class Config for documentation.
  static float uct_c;
  static float initial_root_q_value;
  static float initial_q_penalty;

 private:
  void AppendDebugString(std::ostream& os, const std::string& indent) const;

  /*
  It is slightly confusing which information in a node
  is viewed from the perspective of the player making the
  move, and which is viewed from the perspective of the
  player whose turn it is after the move was made. (Which in
  Flagz may be the same player, if the opponent has no moves
  left.)

  After a single SelfplayRun of a new game, the search
  tree might look like this:

  +------------------------------------+
  | @root
  |------------------------------------|
  | parent_ = nullptr
  | turn_ = 0  // P0 makes 1st move
  | move_ = {}  // No move
  | prior_ = 0.0  // No prior
  | wins_ = 0.0  // Not set
  | value_ = predicted value for turn_
  | children_ = [@c1, ...]
  | NextTurn() = 0
  | MoveTurn() = undefined
  +------------------------------------+

  +------------------------------------
  | @c1
  |------------------------------------
  | parent_ = @root
  | turn_ = 1  // P1 makes 2nd move
  | move_ = Move::Flag(0, 0)
  | prior_ = 0.02 // predicted from @root
  | wins_ = not yet set
  | value_ = not yet set
  | children_ = []
  | NextTurn() = 1
  | MoveTurn() = 0  // parent_->turn_
  +------------------------------------

    Values interpreted as "before move":
    * turn_
        * The player whose turn it is to make the move.
          Should be accessed using the NextTurn() method.
          The MoveTurn() method yields the player whose turn it is to make the
          move.
    * prior_
        The probability to make this move, compared to its siblings.

    Values interpreted as "after move":
    * wins_
        * the accumulated results (or predicted results)
          after the move was made.
          *** We store this value from the perspective of MoveTurn()! ***
    * value_
        * the model's evaluation of the board after the move was made.
        * The model outputs this value from the perspective of the
          player that will make the next move, i.e. NextTurn(),
          because this value is predicted together with the
          move probabilities ("policy") for the NEXT move.
  */

  Node* parent_;
  // The player whose turn it is to make the move stored in this node.
  int turn_;
  // The move to be made.
  Move move_;
  // The prior move probability of this node (relative to its siblings),
  // as predicted by the model. Possibly includes Dirichlet noise.
  float prior_ = 0.0;
  // The accumulated (predicted or actual) results obtained from playing
  // this move, evaluated from the perspective of the MoveTurn() player.
  float wins_ = 0.0;
  // Number of times this node was visited during MCTS.
  int visit_count_ = 0;
  // The model's predicted value of the board after the move_ was made,
  // from the perspective of the NextTurn() player.
  float value_ = 0.0;
  // True if this is a terminal node of the game.
  bool terminal_ = false;
  // Child nodes.
  std::vector<std::unique_ptr<Node>> children_;
};

// Writes the substree starting at root to path as a GraphViz .dot file.
absl::Status WriteDotGraph(const Node& root, const std::string& path);

class PlayoutRunner {
 public:
  struct Stats {
    float result_sum = 0;
    int runs = 0;
    void Add(float result) noexcept;
    float Avg() const noexcept { return runs > 0 ? result_sum / runs : 0; }
    Stats& Merge(const Stats& other) {
      runs += other.runs;
      result_sum += other.result_sum;
      return *this;
    }
    std::string DebugString() const noexcept;
  };

  virtual Stats Run(const Board& board, int turn, int runs) = 0;
  virtual const Stats& AggregatedStats() const = 0;
  virtual void ResetStats() = 0;
  virtual ~PlayoutRunner() = default;
};

class RandomPlayoutRunner : public PlayoutRunner {
 public:
  Stats Run(const Board& board, int turn, int runs) override;
  const Stats& AggregatedStats() const override { return aggregated_stats_; };
  void ResetStats() override { aggregated_stats_ = Stats{}; };

 private:
  Stats aggregated_stats_;
};

class NeuralMCTS {
 public:
  // The model is not owned. Owners of the NeuralMCTS instance must ensure it
  // outlives this instance.
  NeuralMCTS(Model& model, std::unique_ptr<PlayoutRunner> playout_runner,
             const Config& config);

  absl::StatusOr<std::vector<hexzpb::TrainingExample>> PlayGame(
      Board& board, int max_runtime_seconds);

  // Returns a pair of (N, record_example), where N is the number of iterations
  // for the MCTS run and record_example indicates whether the resulting example
  // should be recorded.
  // This is only relevant in a full PlayGame cycle.
  std::pair<int, bool> NumRuns(int move) const noexcept;

  // Executes a single run of the MCTS algorithm, starting at root.
  // This method should be called during "normal" play, i.e. outside of
  // training.
  bool Run(Node& root, const Board& board);
  // SelfplayRun is a variant of Run that should be used for self-play
  // during training.
  // If add_noise is true, Dirichlet noise will be added to the root node's
  // move probs.
  // If run_playouts is true, the PlayoutRunner will be used in the value
  // computation.
  bool SelfplayRun(Node& root, const Board& b, bool add_noise,
                   bool run_playouts);

  // SuggestMove returns the best move suggestion that the NeuralMCTS algorithm
  // comes up with in think_time_millis milliseconds.
  absl::StatusOr<std::unique_ptr<Node>> SuggestMove(int player,
                                                    const Board& board,
                                                    int think_time_millis);

  int PredictionsCount() const { return predictions_count_; }
  int RandomPlayoutsCount() const { return random_playouts_count_; }

 private:
  Config config_;
  // Not owned.
  Model& model_;
  // Used for refining model predictions with random playouts.
  std::unique_ptr<PlayoutRunner> playout_runner_;

  // Stats
  int predictions_count_ = 0;
  int random_playouts_count_ = 0;
};

}  // namespace hexz
