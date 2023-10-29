#include <cpr/cpr.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <ctime>
#include <iostream>

#include "hexz.pb.h"
#include "board.h"

namespace hexz {

using hexzpb::TrainingExample;

int64_t UnixMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Config {
  std::string test_url;
};

void Run(Config config) {
  const auto started_micros = UnixMicros();

  // Check that cpr works:
  if (config.test_url != "") {
    cpr::Response r = cpr::Get(
        cpr::Url{"https://api.github.com/repos/whoshuu/cpr/contributors"},
        cpr::Authentication{"user", "pass", cpr::AuthMode::BASIC},
        cpr::Parameters{{"anon", "true"}, {"key", "value"}});
    std::cout << "Status: " << r.status_code << "\n";
    std::cout << "content-type: " << r.header["content-type"] << "\n";
    std::cout << r.text << "\n";
  }

  // Check that torch works:
  Board b = Board::RandomBoard();
  assert(b.Flags(0) == 3);
  assert(b.Flags(1) == 3);
  auto zero_score = std::make_pair(0.0f, 0.0f);
  assert(b.Score() == zero_score);
  int player = 0;
  for (int i = 0; i < 10; i++) {
    auto moves = b.NextMoves(player);
    assert(moves.size() > 0);
    b.MakeMove(player, moves[0]);
    player = 1 - player;
  }
  assert(b.Score() != zero_score);
  std::cout << b.Score() << "\n";

  // Check that protobuf works:
  const int64_t duration_micros = UnixMicros() - started_micros;
  TrainingExample example;
  example.set_result(1.0);
  example.set_duration_micros(duration_micros);
  example.set_unix_micros(started_micros);
  std::cout << "Hello, hexz: " << example.DebugString() << "\n";
}

}  // namespace hexz

int main() {
  const char* test_url = std::getenv("HEXZ_TEST_URL");
  auto config = hexz::Config{
      .test_url = test_url != nullptr ? std::string(test_url) : "",
  };
  hexz::Run(config);
  return 0;
}
