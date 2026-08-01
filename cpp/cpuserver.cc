#include "cpuserver.h"

#include <absl/cleanup/cleanup.h>
#include <absl/log/absl_log.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <boost/fiber/all.hpp>

#include <cstdlib>

#include "base.h"
#include "board.h"
#include "flagz_tail.h"
#include "mcts.h"

namespace hexz {

CPUPlayerServiceImpl::CPUPlayerServiceImpl(CPUPlayerServiceConfig config)
    : config_{config},
      model_{config.model_key,
             torch::jit::load(config.model_path, config.device_type),
             config.device_type, config.max_batch_size},
      concurrent_rpc_sem_(config.max_concurrent_requests) {
  Config mcts_config{
      .uct_c = config_.uct_c,
      .initial_root_q_value = config_.initial_root_q_value,
      .initial_q_penalty = config_.initial_q_penalty,
  };
  InitializeFromConfig(mcts_config);
  ABSL_LOG(INFO) << "Initialized neural MCTS search parameters: uct_c="
                 << config_.uct_c
                 << ", initial_root_q_value="
                 << config_.initial_root_q_value
                 << ", initial_q_penalty=" << config_.initial_q_penalty;
}

grpc::Status CPUPlayerServiceImpl::ServerInfo(
    grpc::ServerContext*, const hexzpb::ServerInfoRequest*,
    hexzpb::ServerInfoResponse* response) {
  response->set_server_type(hexzpb::ServerInfoResponse::TYPE_NEURAL_MCTS);
  *response->mutable_model_key() = model_.Key();
  return grpc::Status::OK;
}

absl::StatusOr<hexzpb::SuggestMoveResponse> CPUPlayerServiceImpl::DoSuggestMove(
    const hexzpb::GameEngineState& state, int64_t max_think_time_ms,
    int64_t max_iterations) {
  const auto& pb_board = state.flagz().board();
  auto board = Board::FromProto(pb_board);
  if (!board.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "cannot reconstruct board from proto: ", board.status().message()));
  }

  int turn = pb_board.turn() - 1;  // 0-based vs. 1-based...
  if (turn != 0 && turn != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("invalid turn: ", pb_board.turn()));
  }
  if (config_.max_think_time_ms > 0 &&
      max_think_time_ms > config_.max_think_time_ms) {
    max_think_time_ms = config_.max_think_time_ms;
  }

  Config config{
      // These should not have an effect in SuggestMove, but to be safe, we
      // disable fast moves and Dirichlet noise explicitly.
      .fast_move_prob = 0,
      .uct_c = config_.uct_c,
      .initial_root_q_value = config_.initial_root_q_value,
      .initial_q_penalty = config_.initial_q_penalty,
      .dirichlet_concentration = 0,
  };

  if (config_.tail_solver_max_states > 0 && board->Flags(0) == 0 &&
      board->Flags(1) == 0) {
    const TailResolution tail = ResolveSeparatedTail(
        *board, config_.tail_solver_max_states,
        config_.tail_solver_max_micros);
    if (tail.status == TailResolveStatus::kSolved &&
        std::abs(tail.ScoreMargin()) >=
            config_.tail_solver_min_score_margin &&
        tail.optimal_move[turn].has_value()) {
      const Move& best_move = *tail.optimal_move[turn];
      hexzpb::SuggestMoveResponse response;
      auto& stats = *response.mutable_move_stats();
      stats.set_value(turn == 0 ? tail.Result() : -tail.Result());
      auto& scored_move = *stats.add_moves();
      scored_move.set_row(best_move.r);
      scored_move.set_col(best_move.c);
      scored_move.set_type(hexzpb::Field::NORMAL);
      auto& final_score = *scored_move.add_scores();
      final_score.set_kind(hexzpb::SuggestMoveStats::FINAL);
      final_score.set_score(1.0);

      auto& move = *response.mutable_move();
      move.set_player_num(pb_board.turn());
      move.set_move(pb_board.move());
      move.set_cell_type(hexzpb::Field::NORMAL);
      move.set_row(best_move.r);
      move.set_col(best_move.c);

      ABSL_LOG(INFO) << "Resolved separated CPU-player tail: scores "
                     << tail.optimal_score[0] << "-"
                     << tail.optimal_score[1] << ", move "
                     << best_move.DebugString() << ", states "
                     << tail.solve_states << ", duration "
                     << tail.solve_micros << "us";
      return response;
    }
  }

  NeuralMCTS mcts(model_, config);

  // FiberTorchModel requires callers to hold an entry token while making
  // predictions. Tail-solver responses return above without occupying one.
  auto token = model_.Enter();
  absl::StatusOr<std::unique_ptr<Node>> node;
  try {
    node = mcts.SuggestMove(turn, *board, max_think_time_ms, max_iterations);
  } catch (c10::Error& error) {
    ABSL_LOG(ERROR) << "Exception when calling SuggestMove: " << error.msg();
  }
  if (!node.ok()) {
    return absl::InternalError(
        absl::StrCat("SuggestMove error: ", board.status().ToString()));
  }
  ABSL_CHECK(!(*node)->IsLeaf())
      << "SuggestMove must not return OK if there are no valid moves.";

  hexzpb::SuggestMoveResponse response;
  auto& stats = *response.mutable_move_stats();
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
  auto& move = *response.mutable_move();
  move.set_player_num(pb_board.turn());
  move.set_move(pb_board.move());
  move.set_cell_type(best_move.typ == Move::Typ::kFlag ? hexzpb::Field::FLAG
                                                       : hexzpb::Field::NORMAL);
  move.set_row(best_move.r);
  move.set_col(best_move.c);

  return response;
}

grpc::Status CPUPlayerServiceImpl::SuggestMove(
    grpc::ServerContext*, const hexzpb::SuggestMoveRequest* request,
    hexzpb::SuggestMoveResponse* response) {
  int64_t max_think_time_ms = request->max_think_time_ms();
  int64_t max_iterations = request->max_iterations();
  if (max_think_time_ms <= 0 && max_iterations <= 0) {
    return grpc::Status(
        grpc::INVALID_ARGUMENT,
        absl::StrCat(
            "one of max_think_time_ms or max_iterations must be positive"));
  }
  // Only allow this request if there aren't too many ongoing RPCs already.
  bool allow_request = concurrent_rpc_sem_.try_acquire();
  if (!allow_request) {
    return grpc::Status(grpc::RESOURCE_EXHAUSTED,
                        absl::StrCat("too many active RPCs"));
  }
  absl::Cleanup sem_releaser = [this] { concurrent_rpc_sem_.release(); };

  absl::StatusOr<hexzpb::SuggestMoveResponse> r = DoSuggestMove(
      request->game_engine_state(), max_think_time_ms, max_iterations);
  if (!r.ok()) {
    auto status = static_cast<grpc::StatusCode>(r.status().code());
    return grpc::Status(status, std::string(r.status().message()));
  }
  *response = *std::move(r);
  return grpc::Status::OK;
}

}  // namespace hexz
