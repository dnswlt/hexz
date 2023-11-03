#ifndef __HEXZ_CONFIG_H__
#define __HEXZ_CONFIG_H__

#include <string>

namespace hexz {

struct Config {
  std::string test_url;
  std::string training_server_url;
  // Path to a local model.pt file.
  // Can be used for local runs without the training server.
  std::string local_model_path;
  int runs_per_move = 800;
  int max_moves_per_game = 200;
  // Approximate maximum runtime of the worker.
  // Workers tend to finish their games, so this time may be exceeded.
  int max_runtime_seconds = 60;

  static Config FromEnv();
  std::string String() const;
};

std::string GetEnv(const std::string& name);
int GetEnvAsInt(const std::string& name, int default_value);

int64_t UnixMicros();

}  // namespace hexz
#endif  // __HEXZ_CONFIG_H__
