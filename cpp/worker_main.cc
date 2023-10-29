#include <cpr/cpr.h>
#include <chrono>
#include <ctime>
#include <iostream>
// #include <torch/torch.h>

#include "hexz.pb.h"

namespace hexz {

using hexzpb::TrainingExample;

int64_t unix_micros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void Run() {
  const auto started_micros = unix_micros();

  // Check that cpr works:
  cpr::Response r = cpr::Get(
      cpr::Url{"https://api.github.com/repos/whoshuu/cpr/contributors"},
      cpr::Authentication{"user", "pass", cpr::AuthMode::BASIC},
      cpr::Parameters{{"anon", "true"}, {"key", "value"}});
  std::cout << "Status: " << r.status_code << "\n";
  std::cout << "content-type: " << r.header["content-type"] << "\n";
  std::cout << r.text << "\n";
  
  // Check that torch works:
  // torch::Tensor tensor = torch::rand({2, 3});
  // std::cout << tensor << std::endl;

  // Check that protobuf works:
  const int64_t duration_micros = unix_micros() - started_micros;
  TrainingExample example;
  example.set_result(1.0);
  example.set_duration_micros(duration_micros);
  example.set_unix_micros(started_micros);
  std::cout << "Hello, hexz: " << example.DebugString() << "\n";
}

}  // namespace hexz

int main() {
  hexz::Run();
  return 0;
}
