#include "pyapi.h"

#include <stdexcept>

#include "gtest/gtest.h"
#include "hexz.pb.h"

namespace hexz {

namespace {

TEST(MoveSuggesterTest, LoadModelFound) {
  MoveSuggester ms;
  ms.LoadModel("testdata/scriptmodule.pt");
}

TEST(MoveSuggesterTest, LoadModelNotFound) {
  MoveSuggester ms;
  EXPECT_THROW(ms.LoadModel("testdata/does_not_exist.pt"),
               std::invalid_argument);
}

TEST(MoveSuggesterTest, SuggestMove) {
  MoveSuggester ms;
  ms.LoadModel("testdata/scriptmodule.pt");
  hexzpb::SuggestMoveRequest req;
  // Generate a minimal valid request.
  req.set_max_think_time_ms(50);
  auto* flagz = req.mutable_game_engine_state()->mutable_flagz();
  auto* board = flagz->mutable_board();
  board->set_turn(1);
  board->set_move(0);
  auto* p1r = board->add_resources();
  auto* p2r = board->add_resources();
  for (int i = 0; i < 7; i++) {
    p1r->add_num_pieces(1);
    p2r->add_num_pieces(1);
  }
  for (int i=0; i<105; i++) {
    auto* f = board->add_flat_fields();
  }
  std::string sreq;
  ASSERT_TRUE(req.SerializeToString(&sreq));
  hexzpb::SuggestMoveResponse resp;
  auto sresp = ms.SuggestMove(sreq);
  ASSERT_TRUE(resp.ParseFromString(sresp));
  // First move has to be a flag.
  EXPECT_EQ(resp.move().cell_type(), hexzpb::Field::FLAG);
}

}  // namespace

}  // namespace hexz