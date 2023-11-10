#include "base.h"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <algorithm>

namespace hexz {

TEST(BaseTest, GetEnvAsInt) {
  setenv("TEST_FOO", "123", 1);
  setenv("TEST_FOO_NEG", "-123", 1);
  EXPECT_EQ(GetEnvAsInt("TEST_FOO", 0), 123);
  EXPECT_EQ(GetEnvAsInt("TEST_FOO_NEG", 0), -123);
  EXPECT_EQ(GetEnvAsInt("TEST_UNDEFINED", 42), 42);
}

TEST(BaseTest, GetEnvAsDouble) {
  setenv("TEST_FOO_INT", "123", 1);
  setenv("TEST_FOO_DECIMAL", "123.456", 1);
  setenv("TEST_FOO_NEG", "-123.0", 1);
  setenv("TEST_FOO_STR", "asdf", 1);
  EXPECT_EQ(GetEnvAsDouble("TEST_FOO_INT", 0), 123);
  EXPECT_EQ(GetEnvAsDouble("TEST_FOO_DECIMAL", 0), 123.456);
  EXPECT_EQ(GetEnvAsDouble("TEST_FOO_NEG", 0), -123.0);
  EXPECT_EQ(GetEnvAsDouble("TEST_FOO_STR", 17), 0);
}

TEST(DirichletTest, ValuesInExpectedRange) {
  auto v = internal::Dirichlet(10, 0.3);
  ASSERT_EQ(v.size(), 10);
  float min = *std::min_element(v.begin(), v.end());
  float max = *std::max_element(v.begin(), v.end());
  float sum = std::accumulate(v.begin(), v.end(), 0.0);
  EXPECT_FLOAT_EQ(sum, 1);
  EXPECT_GE(min, 0);
  EXPECT_LE(max, 1);
}

}  // namespace hexz
