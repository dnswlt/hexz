#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server_builder.h>

#include <cmath>
#include <optional>
#include <sstream>
#include <string>

#include "version.h"
#include "cpuserver.h"

ABSL_FLAG(std::string, server_addr, "localhost:50051",
          "address on which to serve");
ABSL_FLAG(std::string, model_path, "./scriptmodule.pt",
          "path to the PyTorch module");
ABSL_FLAG(std::string, model_key, "local:0",
          "optional model key (name:checkpoint) for logging purposes and "
          "ServerInfo responses");
ABSL_FLAG(std::string, device, "cpu", "PyTorch device (cpu, cuda, mps)");
ABSL_FLAG(int64_t, max_think_time_ms, 1000,
          "maximum thinking time for SuggestMove requests");
ABSL_FLAG(double, uct_c, 1.5,
          "weight of the PUCT exploration term");
ABSL_FLAG(double, initial_root_q_value, 0.0,
          "Q value assigned to unvisited root children");
ABSL_FLAG(double, initial_q_penalty, 0.0,
          "penalty subtracted from parent Q for deeper unvisited children");
ABSL_FLAG(int, tail_solver_max_states, 50'000,
          "maximum states per player for exact separated-tail solving; "
          "zero disables the fast path");
ABSL_FLAG(int64_t, tail_solver_max_micros, 50'000,
          "maximum microseconds per player for exact separated-tail solving");
ABSL_FLAG(int, tail_solver_min_score_margin, 5,
          "minimum exact final-score margin for the separated-tail fast path");

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
  hexzpb::ModelKey model_key = ParseModelKey(absl::GetFlag(FLAGS_model_key));
  const double uct_c = absl::GetFlag(FLAGS_uct_c);
  const double initial_root_q_value =
      absl::GetFlag(FLAGS_initial_root_q_value);
  const double initial_q_penalty = absl::GetFlag(FLAGS_initial_q_penalty);
  if (!std::isfinite(uct_c) || uct_c <= 0 ||
      !std::isfinite(initial_root_q_value) || initial_root_q_value < -1 ||
      initial_root_q_value > 1 || !std::isfinite(initial_q_penalty) ||
      initial_q_penalty < 0) {
    ABSL_LOG(ERROR) << "Invalid MCTS parameters: uct_c=" << uct_c
                    << ", initial_root_q_value="
                    << initial_root_q_value
                    << ", initial_q_penalty=" << initial_q_penalty;
    return 1;
  }

  hexz::CPUPlayerServiceConfig config{
      .model_path = absl::GetFlag(FLAGS_model_path),
      .model_key = model_key,
      .max_think_time_ms = absl::GetFlag(FLAGS_max_think_time_ms),
      .max_batch_size = 128,
      .max_concurrent_requests = 128,
      .uct_c = static_cast<float>(uct_c),
      .initial_root_q_value = static_cast<float>(initial_root_q_value),
      .initial_q_penalty = static_cast<float>(initial_q_penalty),
      .tail_solver_max_states =
          absl::GetFlag(FLAGS_tail_solver_max_states),
      .tail_solver_max_micros =
          absl::GetFlag(FLAGS_tail_solver_max_micros),
      .tail_solver_min_score_margin =
          absl::GetFlag(FLAGS_tail_solver_min_score_margin),
  };
  std::string device = absl::GetFlag(FLAGS_device);
  if (device == "cuda") {
    config.device_type = torch::kCUDA;
  } else if (device == "mps") {
    config.device_type = torch::kMPS;
  }
  hexz::CPUPlayerServiceImpl service(config);

  grpc::ServerBuilder builder;
  // gRPC sets SO_REUSEPORT to true by default. This cost me an hour of
  // wondering why a model would not behave the way it should, when
  // in fact I received responses from an old server running on the same port.
  // Deactivate!
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
  
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    ABSL_LOG(ERROR)
        << "Cannot instantiate server. Invalid address or address "
           "already in use? Use [::]:<port> to listen on any interface.";
    return 1;
  }
  ABSL_LOG(INFO) << "Server listening on " << addr << " with model key "
                 << model_key.name() << ":" << model_key.checkpoint();
  server->Wait();
}
