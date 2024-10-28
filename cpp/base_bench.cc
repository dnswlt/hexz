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
    sum += rnd.Uniform();
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

BENCHMARK_MAIN();