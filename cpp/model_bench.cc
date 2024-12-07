#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "model.h"

static void BM_TensorBatchFill_Indexing(benchmark::State& state) {
  const int N = 11 * 11 * 10;
  torch::Tensor batch = torch::zeros({state.range(0), N});
  std::vector<torch::Tensor> tensors;
  for (int i = 0; i < state.range(0); i++) {
    tensors.emplace_back(torch::ones({N}));
  }
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      batch[i] = tensors[i];
    }
  }
  EXPECT_EQ(batch.sum().item<float>(), state.range(0) * N);
}
BENCHMARK(BM_TensorBatchFill_Indexing)->Range(1, 1 << 10);

static void BM_TensorBatchFill_Slice(benchmark::State& state) {
  const int N = 11 * 11 * 10;
  torch::Tensor batch = torch::zeros({state.range(0), N});
  torch::Tensor ones = torch::ones({N});
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      batch.slice(0, i, i + 1) = ones;
    }
  }
  EXPECT_EQ(batch.sum().item<float>(), state.range(0) * N);
}
BENCHMARK(BM_TensorBatchFill_Slice)->Range(1, 1 << 10);

static void BM_TensorBatchFill_Stack(benchmark::State& state) {
  const int N = 11 * 11 * 10;
  std::vector<torch::Tensor> tensors;
  for (int i = 0; i < state.range(0); i++) {
    tensors.emplace_back(torch::ones({N}));
  }
  torch::Tensor batch;
  for (auto _ : state) {
    std::vector<torch::Tensor> v;
    for (int i = 0; i < state.range(0); i++) {
      v.push_back(tensors[i]);
    }
    batch = torch::stack(v);
  }
  EXPECT_EQ(batch.sum().item<float>(), state.range(0) * N);
}
BENCHMARK(BM_TensorBatchFill_Stack)->Range(1, 1 << 10);

static void BM_TensorBatchFill_IndexPut(benchmark::State& state) {
  if (state.range(1) && !torch::cuda::is_available()) {
    state.SkipWithMessage(
        "CUDA not available. Skipping pinned memory benchmark.");
    return;
  }
  const int N = 11 * 11 * 10;
  const bool pin_memory = state.range(1);
  torch::Tensor batch = torch::zeros(
      {state.range(0), N},
      torch::TensorOptions().device(torch::kCPU).pinned_memory(pin_memory));
  torch::Tensor ones = torch::ones({N});
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); i++) {
      batch.index_put_({i}, ones);
    }
  }
  EXPECT_EQ(batch.sum().item<float>(), state.range(0) * N);
}
BENCHMARK(BM_TensorBatchFill_IndexPut)->Ranges({{1, 1 << 10}, {false, true}});

BENCHMARK_MAIN();
