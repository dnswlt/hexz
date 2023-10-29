# Use C++17 at least.
set (CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.10)

project(HexzWorker)

# We use protocol buffers to communicate with the training server.
# Need to use REQUIRED CONFIG to ensure CMake uses the protobuf-config.cmake file
# that comes with the protobuf library, not its own outdated FindProtobuf package:
# https://github.com/protocolbuffers/protobuf/issues/12292#issuecomment-1529680040
find_package(protobuf REQUIRED CONFIG)

message(Protobuf_INCLUDE_DIR="${Protobuf_INCLUDE_DIR}")
message(Protobuf_PROTOC_EXECUTABLE="${Protobuf_PROTOC_EXECUTABLE}")

include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${Protobuf_INCLUDE_DIR}")

add_library(hexzpb hexz.pb.h hexz.pb.cc)
target_link_libraries(hexzpb PUBLIC protobuf::libprotobuf)

# Add the executable
add_executable(worker worker_main.cpp)

target_link_libraries(worker PRIVATE hexzpb)

