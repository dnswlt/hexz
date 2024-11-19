#pragma GCC diagnostic ignored "-Wunused-const-variable"
#include "perfm.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <ratio>

namespace hexz {

namespace {

using ::testing::ElementsAreArray;

// FakeClock is a clock that satisfies std::chrono::is_clock
// and can be used for testing APM<FakeClock>.
class FakeClock {
 public:
  using rep = int64_t;
  using period = std::nano;
  using duration = std::chrono::duration<rep, period>;
  using time_point = std::chrono::time_point<FakeClock>;
  static constexpr bool is_steady = false;
  static time_point now() { return current_time_; }
  static void advance(const duration& d) { current_time_ += d; }

 private:
  static inline time_point current_time_{time_point{duration{0}}};
};

#ifndef __clang__
// Apple clang version 16.0.0 (clang-1600.0.26.4) does not support is_clock_v from C++20. 
static_assert(std::chrono::is_clock_v<FakeClock>,
              "FakeClock does not satisfy std::chrono::is_clock!");
#endif

TEST(FakeClockTest, Increment) {
  auto t1 = FakeClock::now();
  FakeClock::advance(std::chrono::seconds(10));
  auto t2 = FakeClock::now();
  EXPECT_EQ(t2 - t1, std::chrono::seconds(10));
  FakeClock::advance(std::chrono::milliseconds(100));
  auto t3 = FakeClock::now();
  EXPECT_EQ(t3 - t1, std::chrono::milliseconds(10100));
}

TEST(APMTest, IncrementRate) {
  APM<FakeClock> apm("/test", 5);
  auto t_start = apm.t_start();
  FakeClock::advance(std::chrono::seconds(2));
  apm.Increment(1);
  apm.Increment(1);
  apm.Rate(1);
  EXPECT_EQ(apm.t_start(), t_start);
  // Expect 2x as many elements as the window size.
  std::vector<int64_t> expected_counts = {
      0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_THAT(apm.CountsForTesting(), ElementsAreArray(expected_counts));
  EXPECT_EQ(apm.Rate(1), 2);
  EXPECT_EQ(apm.Rate(2), 1);
  FakeClock::advance(std::chrono::milliseconds(200));
  EXPECT_FLOAT_EQ(apm.Rate(1), 2 / 1.2);
}

TEST(APMTest, RateBeforeIncrement) {
  APM<FakeClock> apm("/test", 5);
  EXPECT_EQ(apm.Rate(5), 0);
  FakeClock::advance(std::chrono::milliseconds(200));
  EXPECT_EQ(apm.Rate(5), 0);
  FakeClock::advance(std::chrono::milliseconds(1000));
  EXPECT_EQ(apm.Rate(5), 0);
}

TEST(APMTest, IncrementExceedWindow) {
  const int window_size = 5;

  APM<FakeClock> apm("/test", window_size);
  auto t_start = apm.t_start();
  FakeClock::advance(std::chrono::milliseconds(10));

  std::vector<int64_t> expected;
  for (int i = 0; i < window_size * 2; i++) {
    apm.Increment(i);
    expected.push_back(i);
    FakeClock::advance(std::chrono::seconds(1));
  }
  EXPECT_THAT(apm.CountsForTesting(), ElementsAreArray(expected));

  // Now the window needs to be realigned:
  apm.Increment(1);
  EXPECT_THAT(apm.CountsForTesting(), ElementsAreArray(std::vector<int64_t>{
                                          5, 6, 7, 8, 9, 1, 0, 0, 0, 0}));

  // Start time should have been adjusted:
  EXPECT_EQ(apm.t_start() - t_start, std::chrono::seconds(5));

  // Advance one more time, should not realign:
  FakeClock::advance(std::chrono::seconds(1));
  apm.Increment(2);
  EXPECT_THAT(apm.CountsForTesting(), ElementsAreArray(std::vector<int64_t>{
                                          5, 6, 7, 8, 9, 1, 2, 0, 0, 0}));
}

}  // namespace

}  // namespace hexz
