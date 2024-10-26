#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server_builder.h>

#include "cpuserver.h"

ABSL_FLAG(std::string, server_addr, "localhost:50051",
          "address on which to serve");
ABSL_FLAG(std::string, model_path, "./scriptmodule.pt",
          "path to the PyTorch module");
ABSL_FLAG(std::string, device, "cpu", "PyTorch device (cpu, cuda, mps)");

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  std::string addr = absl::GetFlag(FLAGS_server_addr);

  hexz::CPUPlayerServiceConfig config{
      .model_path = absl::GetFlag(FLAGS_model_path),
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