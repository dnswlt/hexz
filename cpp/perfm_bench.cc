#include <benchmark/benchmark.h>

#include "perfm.h"

static void BM_APM(benchmark::State& state) {
  for (auto _ : state) {
    hexz::APMExamples().Increment(1);
  }
}
BENCHMARK(BM_APM)->Range(1, 1 << 10);

BENCHMARK_MAIN();
