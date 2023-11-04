#include "base.h"

#include <gtest/gtest.h>
#include <stdlib.h>

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

}  // namespace hexz
