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

}  // namespace
}  // namespace hexz