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
  // A variation of KataGo's playout cap randomization
  // (https://arxiv.org/abs/1902.10565, section 3.1):
  // With a probability of fast_move_prob, we make a "fast move" with fewer
  // MCTS iterations. This is only done from move 6 onwards (i.e. after all
  // flags have potentially been played). Fast moves are not recorded as
  // examples. The hope is that by playing faster and thus generating more final
  // results, the value predictions can improve faster.
  int runs_per_fast_move = 100;
  // Value between 0 and 1. Set to 0 to disable fast moves.
  float fast_move_prob = 0.0;
  // The runs for move N are calculated as
  //    max(1, runs_per_move * (1 + runs_per_move_gradient * N))
  // The idea is that the first couple of moves are much more important
  // than the "end game" in hexz. The defaul value -0.01 will result
  // in 50% of the original runs_per_move at move 50.
  float runs_per_move_gradient = -0.01;
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
  // Flagz has 18 moves on average (with high variance;
  // there are 85 initial flag moves), so set this to 10/18 ~= 0.55
  // to imitate AlphaZero:
  // https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper
  // or 0.3 to use the setting that was used for chess.
  float dirichlet_concentration = 0;

  // Maximum delay at startup before generating and sending examples.
  // The delay will be uniformly randomized between 0 and startup_delay_seconds.
  // The idea is to avoid a "thundering herd" of workers delivering results
  // to the training server at the same time.
  float startup_delay_seconds = 0.0;

  static Config FromEnv();
  std::string String() const;
};

std::string GetEnv(const std::string& name, const std::string& default_value);
int GetEnvAsInt(const std::string& name, int default_value);
float GetEnvAsFloat(const std::string& name, float default_value);

int64_t UnixMicros();

namespace internal {
extern thread_local std::mt19937 rng;

// Returns a random number in the interval [0, 1] drawn from a uniform
// distribution.
float UnitRandom();

// Libtorch does not have a Dirichlet (or any other nontrivial) distribution yet
// :( So let's roll our own, based on the gamma distribution:
// https://en.wikipedia.org/wiki/Dirichlet_distribution#Related_distributions
std::vector<float> Dirichlet(int n, float concentration);

// Lets the calling thread sleep for a random amount of time between
// [0..max_delay_seconds] seconds.
void RandomDelay(float max_delay_seconds);

}  // namespace internal

}  // namespace hexz
#endif  // __HEXZ_CONFIG_H__
