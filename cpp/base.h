#ifndef __HEXZ_CONFIG_H__
#define __HEXZ_CONFIG_H__

#include <random>
#include <string>

namespace hexz {

struct Config {
  // URL of the training server, e.g. "http://localhost:8080".
  std::string training_server_url;
  // Path to a local model.pt file.
  // Can be used for local runs without the training server.
  std::string local_model_path;
  // MCTS runs executed for each move. Can be further influenced by
  // runs_per_move_decay.
  int runs_per_move = 800;
  // The runs for move N are calculated as
  //    max(1, runs_per_move * (1 + runs_per_move_gradient * N))
  // The idea is that the first couple of moves are much more important
  // than the "end game" in hexz. The defaul value -0.01 will result
  // in 50% of the original runs_per_move at move 50.
  double runs_per_move_gradient = -0.01;
  // Maximum number of moves to make per game before aborting.
  // Only useful for testing, otherwise leave it at the default 200,
  // which ensures games are played till the end.
  int max_moves_per_game = 200;
  // Approximate maximum runtime of the worker.
  int max_runtime_seconds = 60;
  // Maximum number of games the worker should play. Mostly for testing.
  int max_games = -1;
  // Weight of the exploration term in the Puct formula.
  float uct_c = 5.0;
  // Concentration factor ("alpha") of the Dirichlet noise that gets
  // added to the root nodes during MCTS search.
  float dirichlet_concentration = 0;

  static Config FromEnv();
  std::string String() const;
};

std::string GetEnv(const std::string& name);
int GetEnvAsInt(const std::string& name, int default_value);
double GetEnvAsDouble(const std::string& name, double default_value);

int64_t UnixMicros();

namespace internal {
extern thread_local std::mt19937 rng;

// Libtorch does not have a Dirichlet (or any other nontrivial) distribution yet
// :( So let's roll our own, based on the gamma distribution:
// https://en.wikipedia.org/wiki/Dirichlet_distribution#Related_distributions
std::vector<float> Dirichlet(int n, float concentration);

}  // namespace internal

}  // namespace hexz
#endif  // __HEXZ_CONFIG_H__
