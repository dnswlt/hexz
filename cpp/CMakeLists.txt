# Use C++20 at least.
set (CMAKE_CXX_STANDARD 20)

# Build static libraries by default.
# Shared libs seem to be the default for CMake with g++/Linux,
# and that does not play well with static Boost libraries.
set(BUILD_SHARED_LIBS OFF)

cmake_minimum_required(VERSION 3.25.2)

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
# Link Boost libraries statically.
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost CONFIG REQUIRED COMPONENTS fiber)

#
# libtorch
#
# See README.md for instructions on installing libtorch.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#
# GoogleTest
#
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable testing for Google Benchmark" FORCE)
FetchContent_MakeAvailable(googletest)

#
# Google benchmark
#
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.0
)
FetchContent_MakeAvailable(googlebenchmark)

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
add_library(hexzpb hexz.pb.cc)
target_link_libraries(hexzpb PUBLIC protobuf::libprotobuf)

add_library(hexzpb_grpc hexz.grpc.pb.cc)
target_link_libraries(hexzpb_grpc PUBLIC hexzpb protobuf::libprotobuf gRPC::grpc++ gRPC::grpc++_reflection)

add_library(health_grpc health.grpc.pb.cc health.pb.cc health.cc)
target_link_libraries(health_grpc PRIVATE protobuf::libprotobuf gRPC::grpc++ gRPC::grpc++_reflection absl::log)

add_library(grpc_client grpc_client.cc)
target_link_libraries(grpc_client PUBLIC hexzpb_grpc hexzpb absl::log absl::statusor "${TORCH_LIBRARIES}")

add_library(base base.cc perfm.cc queue.h batch.h)
target_link_libraries(base PUBLIC absl::statusor absl::strings absl::base absl::log)

add_library(model model.cc)
target_link_libraries(model PUBLIC base hexzpb Boost::fiber "${TORCH_LIBRARIES}")

add_library(mcts base.cc board.cc mcts.cc perfm.cc)
target_link_libraries(mcts PUBLIC base hexzpb model absl::statusor absl::strings absl::log)

add_executable(worker worker.cc worker_main.cc)
target_link_libraries(worker PRIVATE grpc_client health_grpc mcts absl::log absl::flags absl::flags_parse absl::flags_usage)

add_library(cpuserver cpuserver.cc)
target_link_libraries(cpuserver PUBLIC mcts absl::log hexzpb_grpc)

add_executable(cpuserver_main cpuserver_main.cc)
target_link_libraries(cpuserver_main PRIVATE cpuserver absl::flags absl::flags_parse absl::flags_usage)
set_target_properties(cpuserver_main PROPERTIES OUTPUT_NAME "cpuserver")

# Benchmark for the different GPU pipelines.
add_executable(gpubench gpubench_main.cc)
target_link_libraries(gpubench PRIVATE mcts absl::log absl::flags absl::flags_parse absl::flags_usage grpc_client)

# Experiment: play around with Coroutines.
# For now, fibers seem to do the job just as well.
add_executable(coro coro_main.cc)
target_link_libraries(coro PRIVATE mcts absl::log grpc_client)

#
# Testing
#
enable_testing()

add_executable(board_test board_test.cc)
add_executable(batch_test batch_test.cc)
add_executable(mcts_test mcts_test.cc)
add_executable(model_test model_test.cc)
add_executable(base_test base_test.cc)
add_executable(perfm_test perfm_test.cc)
add_executable(cpuserver_test cpuserver_test.cc)

target_link_libraries(mcts_test GTest::gmock_main mcts)
target_link_libraries(model_test GTest::gmock_main model)
target_link_libraries(board_test GTest::gmock_main mcts)
target_link_libraries(batch_test GTest::gmock_main base)
target_link_libraries(base_test GTest::gtest_main base)
target_link_libraries(perfm_test GTest::gmock_main base)
target_link_libraries(cpuserver_test GTest::gtest_main GTest::gmock cpuserver)

include(GoogleTest)
gtest_discover_tests(mcts_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(model_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(board_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(batch_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(base_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(perfm_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
gtest_discover_tests(cpuserver_test WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

#
# Benchmarking
#
add_executable(base_bench base_bench.cc)
target_link_libraries(base_bench base benchmark::benchmark absl::random_random)

add_executable(perfm_bench perfm_bench.cc)
target_link_libraries(perfm_bench base benchmark::benchmark)
