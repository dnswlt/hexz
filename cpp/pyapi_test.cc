#include "pyapi.h"

#include <stdexcept>

#include "gtest/gtest.h"

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

}  // namespace

}  // namespace hexz