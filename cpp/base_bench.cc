#include <absl/random/random.h>
#include <benchmark/benchmark.h>

#include <cassert>
#include <iostream>

#include "base.h"

static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state) std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

static void BM_Xoshiro256Plus(benchmark::State& state) {
  hexz::internal::Xoshiro256Plus rnd;
  double sum = 0;
  for (auto _ : state) {
    double d = rnd.Uniform();
    if (d > 0.5) {
      sum += d;
    }
  }
  assert(sum > 0);
}
BENCHMARK(BM_Xoshiro256Plus);

static void BM_AbseilUniform(benchmark::State& state) {
  absl::BitGen bitgen;
  double sum = 0;
  for (auto _ : state) {
    sum += absl::Uniform(bitgen, 0.0, 1.0);
  }
  assert(sum > 0);
}
BENCHMARK(BM_AbseilUniform);

static void BM_UnitRandom(benchmark::State& state) {
  float sum = 0;
  for (auto _ : state) {
    sum += hexz::internal::UnitRandom();
  }
  assert(sum > 0);
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