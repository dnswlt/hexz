#include "cpuserver.h"

#include <gmock/gmock.h>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include "hexz.grpc.pb.h"

namespace hexz {

namespace {

using ::testing::Eq;

class CPUPlayerServiceImplTest : public ::testing::Test {
 protected:
  CPUPlayerServiceImplTest()
      : service_(CPUPlayerServiceConfig{
            .model_path = "testdata/scriptmodule.pt",
            .model_key = {},
        }) {
    grpc::ServerBuilder builder;
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    channel_ = server_->InProcessChannel(grpc::ChannelArguments());
    stub_ = hexzpb::CPUPlayerService::NewStub(channel_);
  }

  ~CPUPlayerServiceImplTest() { server_->Shutdown(); }

  std::unique_ptr<hexzpb::CPUPlayerService::Stub> stub_;
  CPUPlayerServiceImpl service_;
  std::unique_ptr<grpc::Server> server_;
  std::shared_ptr<grpc::Channel> channel_;
};

hexzpb::Board DummyProtoBoard() {
  hexzpb::Board board;
  board.set_turn(1);
  for (int i = 0; i < 2; i++) {
    auto& res = *board.add_resources();
    const int iFlag = static_cast<int>(hexzpb::Field::FLAG);
    std::vector<int32_t> num_pieces(iFlag + 1, 0);
    num_pieces[iFlag] = 3;
    res.mutable_num_pieces()->Add(num_pieces.begin(), num_pieces.end());
  }
  for (int i = 0; i < 105; i++) {
    auto& f = *board.add_flat_fields();
    if (i < 10) {
      // Place a rock.
      f.set_type(hexzpb::Field::ROCK);
      f.set_blocked(1 | 2);
    } else if (i < 15) {
      // Place a grass cell.
      f.set_type(hexzpb::Field::GRASS);
      f.set_blocked(1 | 2);
    }
  }
  return board;
}

hexzpb::Board SeparatedTailProtoBoard(int turn, int chain_player) {
  hexzpb::Board board;
  board.set_turn(turn);
  const int iFlag = static_cast<int>(hexzpb::Field::FLAG);
  for (int i = 0; i < 2; ++i) {
    auto& res = *board.add_resources();
    std::vector<int32_t> num_pieces(iFlag + 1, 0);
    res.mutable_num_pieces()->Add(num_pieces.begin(), num_pieces.end());
  }
  for (int i = 0; i < 105; ++i) {
    auto& field = *board.add_flat_fields();
    field.set_type(hexzpb::Field::NORMAL);
    // chain_player owns a forced 1-2-3 chain in cells (0,0)..(0,2);
    // the other player has no reachable cells. Everything else is blocked
    // for both players.
    field.set_blocked(i < 3 ? 1 << (1 - chain_player) : 3);
    field.add_next_val(chain_player == 0 && i == 0 ? 1 : 0);
    field.add_next_val(chain_player == 1 && i == 0 ? 1 : 0);
  }
  return board;
}

TEST_F(CPUPlayerServiceImplTest, SmokeTest) {
  grpc::ClientContext context;
  hexzpb::SuggestMoveRequest request;
  request.set_max_iterations(10);
  auto& flagz = *request.mutable_game_engine_state()->mutable_flagz();
  *flagz.mutable_board() = DummyProtoBoard();
  hexzpb::SuggestMoveResponse response;
  grpc::Status status = stub_->SuggestMove(&context, request, &response);
  EXPECT_THAT(status.error_code(), Eq(grpc::OK));
  EXPECT_TRUE(response.has_move());
  EXPECT_TRUE(response.has_move_stats());
  EXPECT_THAT(response.move().player_num(), Eq(1));
  EXPECT_THAT(response.move().cell_type(), Eq(hexzpb::Field::FLAG));
  EXPECT_EQ(response.move_stats().moves_size(), 90); // 90 valid moves.
}

TEST_F(CPUPlayerServiceImplTest, SeparatedTailBypassesMCTS) {
  grpc::ClientContext context;
  hexzpb::SuggestMoveRequest request;
  request.set_max_iterations(10);
  auto& flagz = *request.mutable_game_engine_state()->mutable_flagz();
  *flagz.mutable_board() = SeparatedTailProtoBoard(/*turn=*/1,
                                                   /*chain_player=*/0);
  hexzpb::SuggestMoveResponse response;

  grpc::Status status = stub_->SuggestMove(&context, request, &response);

  ASSERT_THAT(status.error_code(), Eq(grpc::OK));
  ASSERT_TRUE(response.has_move());
  EXPECT_THAT(response.move().player_num(), Eq(1));
  EXPECT_THAT(response.move().cell_type(), Eq(hexzpb::Field::NORMAL));
  EXPECT_THAT(response.move().row(), Eq(0));
  EXPECT_THAT(response.move().col(), Eq(0));
  ASSERT_TRUE(response.has_move_stats());
  EXPECT_THAT(response.move_stats().value(), Eq(1));
  ASSERT_THAT(response.move_stats().moves_size(), Eq(1));
  EXPECT_THAT(response.move_stats().moves(0).scores(0).score(), Eq(1));
}

TEST_F(CPUPlayerServiceImplTest, SeparatedTailUsesPlayerTwoPerspective) {
  grpc::ClientContext context;
  hexzpb::SuggestMoveRequest request;
  request.set_max_iterations(10);
  auto& flagz = *request.mutable_game_engine_state()->mutable_flagz();
  *flagz.mutable_board() = SeparatedTailProtoBoard(/*turn=*/2,
                                                   /*chain_player=*/1);
  hexzpb::SuggestMoveResponse response;

  grpc::Status status = stub_->SuggestMove(&context, request, &response);

  ASSERT_THAT(status.error_code(), Eq(grpc::OK));
  ASSERT_TRUE(response.has_move());
  EXPECT_THAT(response.move().player_num(), Eq(2));
  EXPECT_THAT(response.move().cell_type(), Eq(hexzpb::Field::NORMAL));
  EXPECT_THAT(response.move().row(), Eq(0));
  EXPECT_THAT(response.move().col(), Eq(0));
  ASSERT_TRUE(response.has_move_stats());
  // The exact P0 result is -1, but response values use the current player's
  // perspective, so winning P1 receives +1.
  EXPECT_THAT(response.move_stats().value(), Eq(1));
  ASSERT_THAT(response.move_stats().moves_size(), Eq(1));
}

}  // namespace
}  // namespace hexz
