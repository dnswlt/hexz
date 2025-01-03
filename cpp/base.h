#pragma once

#include <absl/status/statusor.h>

#include <random>
#include <string>

namespace hexz {

struct Config {
  // gRPC address of the training server, e.g. "localhost:8080".
  std::string training_server_addr;
  // The device on which model predictions are made. Must be one of
  // {"cpu", "mps", "cuda"}.
  std::string device = "cpu";
  // How many threads to use for self-play.
  int worker_threads = 1;
  // Number of fibers to run per thread. Each fiber self-plays games
  // independently.
  // Leave this at 0 to not use fibers at all.
  int fibers_per_thread = 0;
  // Batch size to use in multi-threaded mode for GPU model predictions.
  int prediction_batch_size = 1;
  // MCTS runs executed for each move.
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
  // Maximum number of moves to make per game before aborting.
  // Only useful for testing, otherwise leave it at the default 200,
  // which ensures games are played till the end.
  int max_moves_per_game = 200;
  // Approximate maximum runtime of the worker.
  int max_runtime_seconds = 60;
  // Maximum number of games the worker should play. Mostly for testing.
  int max_games = -1;
  // Weight of the exploration term in the Puct formula.
  float uct_c = 1.5;
  // The initial Q-value for the root node. This is relevant for its children,
  // which compute their initial Q-value based on the Q-value of their parent.
  // (Known as "first play urgency" (FPU) in the literature.)
  // There shouldn't be a need to tweak this value outside tests.
  float initial_root_q_value = -0.2;
  // Value subtracted from the parent's Q value to calculate an unvisited
  // child's initial Q value.
  float initial_q_penalty = 0.3;
  // Concentration factor ("alpha") of the Dirichlet noise that gets
  // added to the root nodes during MCTS search.
  // Flagz has 18 moves on average (with high variance;
  // there are 85 initial flag moves), so set this to 10/18 ~= 0.55
  // to imitate AlphaZero:
  // https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper
  // or 0.3 to use the setting that was used for chess.
  // Set this to zero to disable Dirichlet noise.
  float dirichlet_concentration = 0;
  // Number of random playouts to play at each leaf node to improve the
  // (initially totally useless) model value predictions. Random playouts
  // are also used to resign early.
  // Set this to zero to disable random playouts.
  int random_playouts = 0;
  // Threshold Q value above which a game will be resigned, to speed up
  // self-play. Currently this is only used for random playouts.
  float resign_threshold = std::numeric_limits<float>::max();
  // Maximum delay at startup before generating and sending examples.
  // The delay will be uniformly randomized between 0 and startup_delay_seconds.
  // The idea is to avoid a "thundering herd" of workers delivering results
  // to the training server at the same time.
  float startup_delay_seconds = 0.0;

  // Set this to true to use pthread thread affinity to pin each thread to a
  // dedicated core. `worker_threads` must be less than or equal to the number
  // of online cores (as reported by sysconf(_SC_NPROCESSORS_ONLN)). Only
  // available on Linux-based systems.
  bool pin_threads = false;
  // Set this to true to have /proc/self/status logged every N seconds.
  bool debug_memory_usage = false;
  // Set this to true to enable the gRPC Health service.
  bool enable_health_service = false;
  // Suspend the worker when the training server signals that it is training.
  // This is only useful when worker and training server run on the same
  // machine.
  bool suspend_while_training = false;

  // Set this to true to avoid sending examples to training server.
  // Useful if you want to test the worker against a real model, but
  // not taint it with low-quality examples.
  bool dry_run = false;

  static absl::StatusOr<Config> FromEnv();
  std::string String() const;
};

std::string GetEnv(const std::string& name, const std::string& default_value);
int GetEnvAsInt(const std::string& name, int default_value);
float GetEnvAsFloat(const std::string& name, float default_value);
bool GetEnvAsBool(const std::string& name, bool default_value);

int64_t UnixMicros();

// Returns a 64-bit random hex string (ex: "dead1337")
std::string RandomUid();

// Helper to implement RAII-style scope guards.
// A ScopeGuard executes the function provided as a c'tor arg
// once the guard object goes out of scope.
class ScopeGuard {
 public:
  explicit ScopeGuard(std::function<void()> cleanup)
      : active_(true), cleanup_(std::move(cleanup)) {}
  ScopeGuard(const ScopeGuard& other) = delete;
  ScopeGuard(ScopeGuard&& other)
      : active_(other.active_), cleanup_(std::move(other.cleanup_)) {
    other.Dismiss();
  }
  ScopeGuard& operator=(const ScopeGuard& other) = delete;
  ScopeGuard& operator=(ScopeGuard&& other) {
    if (this != &other) {
      cleanup_ = std::move(other.cleanup_);
      active_ = other.active_;
      other.Dismiss();
    }
    return *this;
  }
  ~ScopeGuard() {
    if (active_) {
      cleanup_();
    }
  }
  void Dismiss() { active_ = false; }

 private:
  bool active_;
  std::function<void()> cleanup_;
};

namespace internal {
extern thread_local std::mt19937 rng;

// xoshiro256+ for fast uniform random numbers in the [0, 1) interval.
// https://prng.di.unimi.it/
class Xoshiro256Plus {
 public:
  // "Implement" UniformRandomBitGenerator, so that Xoshiro256Plus can be used
  // by std::uniform_int_distribution and friends.
  using result_type = uint64_t;
  static constexpr result_type min() {
    return std::numeric_limits<result_type>::min();
  }
  static constexpr result_type max() {
    return std::numeric_limits<result_type>::max();
  }
  result_type operator()() { return Next(); }

  Xoshiro256Plus() {
    std::uniform_int_distribution<uint64_t> dis{
        0, std::numeric_limits<uint64_t>::max()};
    for (size_t i = 0; i < std::size(s); i++) {
      s[i] = dis(rng);
    }
  }

  // Returns a random number in the interval [0, 1) drawn from a uniform
  // distribution.
  double Uniform() {
    uint64_t x = Next();
    return (x >> 11) * 0x1.0p-53;
  }

#if defined(__SIZEOF_INT128__)
  // Returns a random number in the open interval [0, n) drawn from a uniform
  // distribution.f
  int Intn(int n) {
    uint64_t r = Next();
    __uint128_t p = static_cast<__uint128_t>(r) * static_cast<__uint128_t>(n);
    return static_cast<uint64_t>(p >> 64);  // Take the upper 64 bits
  }
#else
  int Intn(int n) {
    std::uniform_int_distribution<int> dis{0, n};
    return dis(*this);
  }
#endif

  // Libtorch does not have a Dirichlet (or any other nontrivial) distribution
  // yet :( So let's roll our own, based on the gamma distribution:
  // https://en.wikipedia.org/wiki/Dirichlet_distribution#Related_distributions
  std::vector<float> Dirichlet(int n, float concentration) {
    std::gamma_distribution<float> gamma(concentration, 1.0);
    std::vector<float> v(n);
    float sum = 0;
    for (int i = 0; i < n; i++) {
      v[i] = gamma(*this);
      sum += v[i];
    }
    for (int i = 0; i < n; i++) {
      v[i] /= sum;
    }
    return v;
  }

 private:
  inline uint64_t Next() {
    const uint64_t result = s[0] + s[3];

    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = Rotl(s[3], 45);

    return result;
  }

  inline uint64_t Rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  uint64_t s[4];
};

// Alias for the random number generator to use.
using RNG = Xoshiro256Plus;

}  // namespace internal

}  // namespace hexz
