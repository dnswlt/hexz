#include "base.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdlib.h>

#include <algorithm>

namespace hexz {

using ::testing::MatchesRegex;

TEST(BaseTest, GetEnvAsInt) {
  setenv("TEST_FOO", "123", 1);
  setenv("TEST_FOO_NEG", "-123", 1);
  EXPECT_EQ(GetEnvAsInt("TEST_FOO", 0), 123);
  EXPECT_EQ(GetEnvAsInt("TEST_FOO_NEG", 0), -123);
  EXPECT_EQ(GetEnvAsInt("TEST_UNDEFINED", 42), 42);
}

TEST(BaseTest, GetEnvAsFloat) {
  setenv("TEST_FOO_INT", "123", 1);
  setenv("TEST_FOO_DECIMAL", "123.25", 1);
  setenv("TEST_FOO_NEG", "-123.0", 1);
  setenv("TEST_FOO_STR", "asdf", 1);
  EXPECT_EQ(GetEnvAsFloat("TEST_FOO_INT", 0), 123);
  EXPECT_EQ(GetEnvAsFloat("TEST_FOO_DECIMAL", 0), 123.25);
  EXPECT_EQ(GetEnvAsFloat("TEST_FOO_NEG", 0), -123.0);
  EXPECT_EQ(GetEnvAsFloat("TEST_FOO_STR", 17), 0);
}

TEST(BaseTest, DefaultConfig) {
  Config def{};
  EXPECT_EQ(def.max_games, -1);
}

TEST(BaseTest, RandomUid) {
  std::string uid = RandomUid();
  EXPECT_EQ(uid.size(), 8);
  EXPECT_THAT(uid, MatchesRegex("[0-9a-f]+"));
}

TEST(DirichletTest, ValuesInExpectedRange) {
  internal::RNG rng;
  auto v = rng.Dirichlet(10, 0.3);
  ASSERT_EQ(v.size(), 10);
  float min = *std::min_element(v.begin(), v.end());
  float max = *std::max_element(v.begin(), v.end());
  float sum = std::accumulate(v.begin(), v.end(), 0.0);
  EXPECT_FLOAT_EQ(sum, 1);
  EXPECT_GE(min, 0);
  EXPECT_LE(max, 1);
}

TEST(Xoshiro256PlusTest, IsUniform) {
  internal::Xoshiro256Plus rnd;
  constexpr size_t N = 100;
  int hist[N] = {0};
  for (int i = 0; i < N * 100; i++) {
    double x = rnd.Uniform();
    size_t j = static_cast<size_t>(x * N);
    hist[j]++;
  }
  for (int i = 0; i < std::size(hist); i++) {
    EXPECT_GE(hist[i], N / 2);
    EXPECT_LE(hist[i], (N * 3) / 2);
  }
}

TEST(Xoshiro256PlusTest, IntnOne) {
  internal::Xoshiro256Plus rnd;
  EXPECT_EQ(rnd.Intn(1), 0);
}

TEST(Xoshiro256PlusTest, IntnRange) {
  internal::Xoshiro256Plus rnd;
  constexpr size_t N = 10;
  int hist[N] = {0};
  for (int i = 0; i < N * 100; i++) {
    int r = rnd.Intn(N);
    ASSERT_GE(r, 0);
    ASSERT_LT(r, N);
    hist[r]++;
  }
  for (int i = 0; i < std::size(hist); i++) {
    EXPECT_GE(hist[i], 0);
  }
}

TEST(Xoshiro256PlusTest, UsesRandomSeed) {
  internal::Xoshiro256Plus rnd1;
  internal::Xoshiro256Plus rnd2;
  EXPECT_NE(rnd1.Uniform(), rnd2.Uniform());
}

}  // namespace hexz
