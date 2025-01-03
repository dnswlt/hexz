#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server_builder.h>

#include <optional>
#include <sstream>
#include <string>

#include "cpuserver.h"

ABSL_FLAG(std::string, server_addr, "localhost:50051",
          "address on which to serve");
ABSL_FLAG(std::string, model_path, "./scriptmodule.pt",
          "path to the PyTorch module");
ABSL_FLAG(std::string, model_key, "local:0",
          "optional model key (name:checkpoint) for logging purposes");
ABSL_FLAG(std::string, device, "cpu", "PyTorch device (cpu, cuda, mps)");
ABSL_FLAG(int64_t, max_think_time_ms, 1000,
          "maximum thinking time for SuggestMove requests");

hexzpb::ModelKey ParseModelKey(const std::string& input) {
  std::istringstream iss(input);
  std::string name;
  int checkpoint;

  hexzpb::ModelKey key;
  if (std::getline(iss, name, ':') && iss >> checkpoint) {
    *key.mutable_name() = name;
    key.set_checkpoint(checkpoint);
  }
  return key;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  std::string addr = absl::GetFlag(FLAGS_server_addr);

  hexz::CPUPlayerServiceConfig config{
      .model_path = absl::GetFlag(FLAGS_model_path),
      .model_key = ParseModelKey(absl::GetFlag(FLAGS_model_key)),
      .max_think_time_ms = absl::GetFlag(FLAGS_max_think_time_ms),
  };
  std::string device = absl::GetFlag(FLAGS_device);
  if (device == "cuda") {
    config.device = torch::kCUDA;
  } else if (device == "mps") {
    config.device = torch::kMPS;
  }
  hexz::CPUPlayerServiceImpl service(config);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << addr << std::endl;
  server->Wait();
}