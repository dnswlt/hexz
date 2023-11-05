# Use C++17 at least.
set (CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.10)

project(HexzWorker)

include(FetchContent)

#
# cpr
#
# cpr is a nice wrapper around libcurl. We use it to make http requests to
# the training server.
#
# The commit hash is for 1.10.5, found on https://github.com/libcpr/cpr/releases.
FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
                         GIT_TAG 3b15fa82ea74739b574d705fea44959b58142eb8)
FetchContent_MakeAvailable(cpr)

#
# Protocol Buffers
#
# We use protocol buffers to communicate with the training server.
# Need to use REQUIRED CONFIG to ensure CMake uses the protobuf-config.cmake file
# that comes with the protobuf library, not its own outdated FindProtobuf package:
# https://github.com/protocolbuffers/protobuf/issues/12292#issuecomment-1529680040
find_package(protobuf REQUIRED CONFIG)
find_package(absl REQUIRED CONFIG)

#
# libtorch
#
# See README.md for instructions on installing libtorch.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#
# GoogleTest
#
# f8d7d77c06936315286eb55f8de22cd23c188571 is v1.14.
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)
FetchContent_MakeAvailable(googletest)


# Debug output for some vars that were relevant getting this whole mess to work:
message(CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
message(CMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE}")
message(CMAKE_CXX_FLAGS_DEBUG="${CMAKE_CXX_FLAGS_DEBUG}")
message(CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}")
message(CMAKE_FIND_PACKAGE_REDIRECTS_DIR="${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}")

# include_directories("${CMAKE_CURRENT_BINARY_DIR}")

#
# Libraries and main binary
#

add_library(hexzpb hexz.pb.h hexz.pb.cc)
target_link_libraries(hexzpb PUBLIC protobuf::libprotobuf)

add_library(mcts base.cc board.cc mcts.cc perfm.cc rpc.cc)
target_link_libraries(mcts PUBLIC absl::statusor absl::strings absl::log cpr::cpr hexzpb "${TORCH_LIBRARIES}")

add_library(hexz_pyapi pyapi.cc)
target_link_libraries(hexz_pyapi PUBLIC absl::strings absl::log hexzpb mcts)

add_executable(worker worker_main.cc)
target_link_libraries(worker PRIVATE mcts)

#
# Testing
#
enable_testing()

add_executable(board_test board_test.cc)
add_executable(mcts_test mcts_test.cc)
add_executable(base_test base_test.cc)

target_link_libraries(mcts_test GTest::gtest_main mcts)
target_link_libraries(board_test GTest::gtest_main mcts)
target_link_libraries(base_test GTest::gtest_main mcts)

include(GoogleTest)
gtest_discover_tests(mcts_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(board_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(base_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
