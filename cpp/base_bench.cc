#include <absl/random/random.h>
#include <benchmark/benchmark.h>

#include <iostream>

#include "base.h"

static void BM_Xoshiro256Plus(benchmark::State& state) {
  hexz::internal::Xoshiro256Plus rnd;
  double sum = 0;
  for (auto _ : state) {
    double d = rnd.Uniform();
    benchmark::DoNotOptimize(sum += d);
  }
}
BENCHMARK(BM_Xoshiro256Plus);

static void BM_Xoshiro256PlusIntn(benchmark::State& state) {
  hexz::internal::Xoshiro256Plus rnd;
  int sum = 0;
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      int d = rnd.Intn(i);
      benchmark::DoNotOptimize(sum += d);
    }
  }
}
BENCHMARK(BM_Xoshiro256PlusIntn)->Range(1, 1 << 10);

static void BM_XoshiroWithStdlibIntn(benchmark::State& state) {
  hexz::internal::Xoshiro256Plus rnd;
  int sum = 0;
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      std::uniform_int_distribution<int> dis{0, i};
      int d = dis(rnd);
      benchmark::DoNotOptimize(sum += d);
    }
  }
}
BENCHMARK(BM_XoshiroWithStdlibIntn)->Range(1, 1 << 10);

static void BM_MersenneIntn(benchmark::State& state) {
  std::mt19937 rng{std::random_device{}()};
  int sum = 0;
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      std::uniform_int_distribution<int> dis{0, i};
      int d = dis(rng);
      benchmark::DoNotOptimize(sum += d);
    }
  }
}
BENCHMARK(BM_MersenneIntn)->Range(1, 1 << 10);

static void BM_AbseilUniform(benchmark::State& state) {
  absl::BitGen bitgen;
  double sum = 0;
  for (auto _ : state) {
    double d = absl::Uniform(bitgen, 0.0, 1.0);
    benchmark::DoNotOptimize(sum += d);
  }
}
BENCHMARK(BM_AbseilUniform);

static void BM_UnitRandom(benchmark::State& state) {
  std::mt19937 rng{std::random_device{}()};

  float sum = 0;
  for (auto _ : state) {
    std::uniform_real_distribution<float> dis{0, 1.0};
    float d = dis(rng);
    benchmark::DoNotOptimize(sum += d);
  }
}
BENCHMARK(BM_UnitRandom);

static void BM_Xoshiro256PlusShuffle(benchmark::State& state) {
  hexz::internal::Xoshiro256Plus rnd;
  std::vector<int> v(100);
  for (int i = 0; i < v.size(); i++) {
    v[i] = i;
  }
  for (auto _ : state) {
    std::shuffle(v.begin(), v.end(), rnd);
  }
}
BENCHMARK(BM_Xoshiro256PlusShuffle);

static void BM_MersenneShuffle(benchmark::State& state) {
  std::vector<int> v(100);
  for (int i = 0; i < v.size(); i++) {
    v[i] = i;
  }
  for (auto _ : state) {
    std::shuffle(v.begin(), v.end(), hexz::internal::rng);
  }
}
BENCHMARK(BM_MersenneShuffle);

BENCHMARK_MAIN();