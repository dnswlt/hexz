#include "pyapi.h"

#include <absl/log/absl_log.h>
#include <absl/strings/str_format.h>
#include <torch/script.h>

#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "board.h"
#include "hexz.pb.h"
#include "mcts.h"

namespace hexz {

class MoveSuggester::impl {
 public:
  std::string SuggestMove(const std::string& request);
  void LoadModel(const std::string& path);

 private:
  std::unique_ptr<TorchModel> model_;
};

// Must be in the .cc file to make pimpl with std::unique_ptr work.
MoveSuggester::MoveSuggester() : pimpl_{std::make_unique<impl>()} {}
MoveSuggester::MoveSuggester(MoveSuggester&&) = default;
MoveSuggester& MoveSuggester::operator=(MoveSuggester&& other) = default;
MoveSuggester::~MoveSuggester() = default;

std::string MoveSuggester::SuggestMove(const std::string& request) {
  return pimpl().SuggestMove(request);
}

void MoveSuggester::LoadModel(const std::string& path) {
  pimpl().LoadModel(path);
}

std::string MoveSuggester::impl::SuggestMove(const std::string& request) {
  hexzpb::SuggestMoveRequest req;
  if (!req.ParseFromString(request)) {
    throw std::invalid_argument("not a valid SuggestMoveRequest proto");
  }
  if (req.max_think_time_ms() <= 0) {
    throw std::invalid_argument("max_think_time_ms must be positive.");
  }
  if (!req.game_engine_state().has_flagz()) {
    throw std::invalid_argument("SuggestMoveRequest has no flagz state");
  }
  const auto& pb_board = req.game_engine_state().flagz().board();
  ABSL_DLOG(INFO) << "SuggestMove: received request for move:"
                  << pb_board.move() << " turn:" << pb_board.turn();
  auto board = Board::FromProto(pb_board);
  if (!board.ok()) {
    throw std::invalid_argument(
        absl::StrCat("Invalid board: ", board.status().ToString()));
  }
  Config config{
      // Don't limit the runs per move, we only rely on
      // limiting the execution time.
      .runs_per_move = std::numeric_limits<int>::max(),
  };
  NeuralMCTS mcts(*model_, config);
  int turn = pb_board.turn() - 1;  // 0-based vs. 1-based...
  if (turn != 0 && turn != 1) {
    throw std::invalid_argument(
        absl::StrCat("invalid turn: ", pb_board.turn()));
  }
  int64_t t_started = UnixMicros();
  auto node = mcts.SuggestMove(turn, *board, req.max_think_time_ms());
  if (!node.ok()) {
    throw std::runtime_error(
        absl::StrCat("SuggestMove: ", board.status().ToString()));
  }
  ABSL_CHECK(!(*node)->IsLeaf())
      << "SuggestMove must not return OK if there are no valid moves.";
  hexzpb::SuggestMoveResponse resp;
  const auto& cs = (*node)->children();
  int most_visited_idx = 0;
  auto& stats = *resp.mutable_move_stats();
  for (int i = 0; i < cs.size(); i++) {
    const auto& c = cs[i];
    if (c->visit_count() > cs[most_visited_idx]->visit_count()) {
      most_visited_idx = i;
    }
    auto& move = *stats.add_moves();
    move.set_row(c->move().r);
    move.set_col(c->move().c);
    move.set_type(c->move().typ == 0 ? hexzpb::Field::FLAG
                                      : hexzpb::Field::NORMAL);
    auto& final_score = *move.add_scores();
    final_score.set_kind(hexzpb::SuggestMoveStats::FINAL);
    final_score.set_score(float(c->visit_count()) / (*node)->visit_count());
    auto& prior_score = *move.add_scores();
    prior_score.set_kind(hexzpb::SuggestMoveStats::MCTS_PRIOR);
    prior_score.set_score(c->Prior());
  }
  const auto& best_move = cs[most_visited_idx]->move();
  ABSL_DLOG(INFO) << "SuggestMove: computed move suggestion "
                  << best_move.DebugString() << " in "
                  << (UnixMicros() - t_started) / 1000 << "ms";
  auto& move = *resp.mutable_move();
  move.set_player_num(pb_board.turn());
  move.set_move(pb_board.move());
  move.set_cell_type(best_move.typ == 0 ? hexzpb::Field::FLAG
                                        : hexzpb::Field::NORMAL);
  move.set_row(best_move.r);
  move.set_col(best_move.c);
  return resp.SerializeAsString();
}

void MoveSuggester::impl::LoadModel(const std::string& path) {
  ABSL_LOG(INFO) << "Loading model from " << path;
  try {
    std::ifstream f_in(path, std::ios::binary);
    if (!f_in.is_open()) {
      throw new std::invalid_argument("cannot read model from path " + path);
    }
    auto m = torch::jit::load(f_in);
    m.to(torch::kCPU);
    m.eval();
    model_ = std::make_unique<TorchModel>(m);
    ABSL_LOG(INFO)
        << "Model loaded successfully. Ready to serve SuggestMove requests!";

  } catch (const c10::Error& e) {
    throw new std::runtime_error("torch::jit::load(" + path + "): " + e.msg());
  }
}

}  // namespace hexz
