# Use C++20 at least.
set (CMAKE_CXX_STANDARD 20)

cmake_minimum_required(VERSION 3.10)

project(HexzWorker)

include(FetchContent)

#
# Protocol Buffers
#
# We use protocol buffers to communicate with the training server.
# Need to use REQUIRED CONFIG to ensure CMake uses the protobuf-config.cmake file
# that comes with the protobuf library, not its own outdated FindProtobuf package:
# https://github.com/protocolbuffers/protobuf/issues/12292#issuecomment-1529680040
find_package(absl CONFIG REQUIRED)
find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

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

add_library(hexzpb_grpc hexz.grpc.pb.h hexz.grpc.pb.cc)
target_link_libraries(hexzpb_grpc PUBLIC hexzpb protobuf::libprotobuf gRPC::grpc++ gRPC::grpc++_reflection)

add_library(grpc_client grpc_client.h grpc_client.cc)
target_link_libraries(grpc_client PUBLIC hexzpb_grpc hexzpb absl::log absl::statusor "${TORCH_LIBRARIES}")

add_library(mcts base.cc board.cc mcts.cc perfm.cc batch.h)
target_link_libraries(mcts PUBLIC absl::statusor absl::strings absl::log hexzpb "${TORCH_LIBRARIES}")

add_library(hexz_pyapi pyapi.cc)
target_link_libraries(hexz_pyapi PUBLIC absl::strings absl::log hexzpb mcts)

add_executable(worker worker.cc worker_main.cc)
target_link_libraries(worker PRIVATE mcts absl::log absl::flags absl::flags_parse absl::flags_usage grpc_client)

add_executable(coro coro_main.cc)
target_link_libraries(coro PRIVATE mcts absl::log grpc_client)

add_executable(gpubench gpubench_main.cc)
target_link_libraries(gpubench PRIVATE mcts absl::log absl::flags absl::flags_parse absl::flags_usage grpc_client)

# Experiment: Boost and fibers
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS fiber)
add_executable(fiber fiber_main.cc)
target_link_libraries(fiber PRIVATE Boost::fiber)
# End Experiment

#
# Testing
#
enable_testing()

add_executable(board_test board_test.cc)
add_executable(batch_test batch_test.cc)
add_executable(mcts_test mcts_test.cc)
add_executable(base_test base_test.cc)
add_executable(pyapi_test pyapi_test.cc)

target_link_libraries(mcts_test GTest::gmock_main mcts)
target_link_libraries(board_test GTest::gmock_main mcts)
target_link_libraries(batch_test GTest::gmock_main mcts)
target_link_libraries(base_test GTest::gtest_main mcts)
target_link_libraries(pyapi_test GTest::gtest_main hexz_pyapi mcts)

include(GoogleTest)
gtest_discover_tests(mcts_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(board_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(batch_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(base_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(pyapi_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
