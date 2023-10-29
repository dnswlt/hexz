#include <chrono>
#include <ctime>
#include <iostream>

#include "hexz.pb.h"

using hexzpb::TrainingExample;

int main() {
  const auto started = std::chrono::steady_clock::now();
  const int64_t started_micros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          started.time_since_epoch())
          .count();
  const int64_t duration_micros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - started).count();
  TrainingExample example;
  example.set_result(1.0);
  example.set_duration_micros(duration_micros);
  example.set_unix_micros(started_micros);
  std::cout << "Hello, hexz: " << example.DebugString() << "\n";
  return 0;
}
