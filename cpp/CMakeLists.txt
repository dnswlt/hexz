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

#
# libtorch
#
# See README.md for instructions on installing libtorch.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Debug output for some vars that were relevant getting this whole mess to work:
message(CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}")
message(CMAKE_FIND_PACKAGE_REDIRECTS_DIR="${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}")

# include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(hexzpb hexz.pb.h hexz.pb.cc)
target_link_libraries(hexzpb PUBLIC protobuf::libprotobuf)

# Add the executable
add_executable(worker worker_main.cc)

target_link_libraries(worker PRIVATE hexzpb cpr::cpr "${TORCH_LIBRARIES}")
