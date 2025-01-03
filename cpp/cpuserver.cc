#include "cpuserver.h"

#include <absl/log/absl_log.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "base.h"
#include "board.h"
#include "mcts.h"

namespace hexz {

CPUPlayerServiceImpl::CPUPlayerServiceImpl(CPUPlayerServiceConfig config)
    : config_{config},
      model_{config.model_key, torch::jit::load(config.model_path),
             config.device} {}

grpc::Status CPUPlayerServiceImpl::SuggestMove(
    grpc::ServerContext*, const hexzpb::SuggestMoveRequest* request,
    hexzpb::SuggestMoveResponse* response) {
  const auto& pb_board = request->game_engine_state().flagz().board();
  ABSL_LOG(INFO) << "SuggestMove: received request for move:" << pb_board.move()
                 << " turn:" << pb_board.turn();
  auto board = Board::FromProto(pb_board);
  if (!board.ok()) {
    return grpc::Status(grpc::INVALID_ARGUMENT,
                        absl::StrCat("cannot reconstruct board from proto: ",
                                     board.status().message()));
  }

  int turn = pb_board.turn() - 1;  // 0-based vs. 1-based...
  if (turn != 0 && turn != 1) {
    return grpc::Status(grpc::INVALID_ARGUMENT,
                        absl::StrCat("invalid turn: ", pb_board.turn()));
  }

  Config config{
      // These should not have an effect in SuggestMove, but to be safe, we
      // disable fast moves and Dirichlet noise explicitly.
      .fast_move_prob = 0,
      .dirichlet_concentration = 0,
      // Random playouts should only be used during self-play.
      .random_playouts = 0,
  };
  // For now, acquite a lock for single module on each request.
  // We don't intend to let cpuserver serve multiple requests in parallel.
  std::unique_lock<std::mutex> module_lock(module_mut_);

  NeuralMCTS mcts(model_, /*playout_runner=*/nullptr, config);

  int64_t think_time = request->max_think_time_ms();
  int64_t max_iterations = request->max_iterations();
  if (think_time <= 0 && max_iterations <= 0) {
    return grpc::Status(
        grpc::INVALID_ARGUMENT,
        absl::StrCat(
            "one of max_think_time_ms or max_iterations must be positive"));
  }
  if (config_.max_think_time_ms > 0 && think_time > config_.max_think_time_ms) {
    think_time = config_.max_think_time_ms;
  }
  int64_t t_started = UnixMicros();
  absl::StatusOr<std::unique_ptr<Node>> node;
  try {
    node = mcts.SuggestMove(turn, *board, think_time, max_iterations);
  } catch (c10::Error& error) {
    ABSL_LOG(ERROR) << "Exception when calling SuggestMove: " << error.msg();
  }
  if (!node.ok()) {
    return grpc::Status(
        grpc::INTERNAL,
        absl::StrCat("SuggestMove error: ", board.status().ToString()));
  }
  ABSL_CHECK(!(*node)->IsLeaf())
      << "SuggestMove must not return OK if there are no valid moves.";
  auto& stats = *response->mutable_move_stats();
  stats.set_value((*node)->value());
  for (const auto& c : (*node)->children()) {
    auto& move = *stats.add_moves();
    move.set_row(c->move().r);
    move.set_col(c->move().c);
    move.set_type(c->move().typ == Move::Typ::kFlag ? hexzpb::Field::FLAG
                                                    : hexzpb::Field::NORMAL);
    auto& final_score = *move.add_scores();
    final_score.set_kind(hexzpb::SuggestMoveStats::FINAL);
    final_score.set_score(float(c->visit_count()) / (*node)->visit_count());
    auto& prior_score = *move.add_scores();
    prior_score.set_kind(hexzpb::SuggestMoveStats::MCTS_PRIOR);
    prior_score.set_score(c->prior());
  }
  const auto& best_move = (*node)->MostVisitedChild()->move();
  ABSL_LOG(INFO) << "SuggestMove: computed move suggestion "
                 << best_move.DebugString() << " in "
                 << (UnixMicros() - t_started) / 1000 << "ms";
  auto& move = *response->mutable_move();
  move.set_player_num(pb_board.turn());
  move.set_move(pb_board.move());
  move.set_cell_type(best_move.typ == Move::Typ::kFlag ? hexzpb::Field::FLAG
                                                       : hexzpb::Field::NORMAL);
  move.set_row(best_move.r);
  move.set_col(best_move.c);

  return grpc::Status::OK;
}

}  // namespace hexz
