#include "perfm.h"

#include <inttypes.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace hexz {

Perfm::CumulativeStats Perfm::cum_stats_[Perfm::StatsSize];
thread_local Perfm::CumulativeStats Perfm::stats_[Perfm::StatsSize];
std::mutex Perfm::mut_;

void Perfm::Init() {
  for (int i = 0; i < StatsSize; i++) {
    cum_stats_[i].label = static_cast<Perfm::Label>(i);
  }
}

void Perfm::PrintStats() {
  std::unique_lock<std::mutex> lk(Perfm::mut_);
  std::vector<Perfm::CumulativeStats> stats;
  int scope_len = 3;
  for (int i = 0; i < StatsSize; i++) {
    if (Perfm::cum_stats_[i].count == 0) continue;
    stats.push_back(Perfm::cum_stats_[i]);
    auto s = Perfm::LabelName(static_cast<Perfm::Label>(i)).size() + 3;
    if (s > scope_len) {
      scope_len = s;
    }
  }
  // Sort by elapsed time, descending.
  std::sort(stats.begin(), stats.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.elapsed_nanos > rhs.elapsed_nanos;
  });

  std::printf("%-*s %10s %10s %12s\n", scope_len, "scope", "total_time",
              "count", "ops/s");

  for (const auto& s : stats) {
    std::printf("%-*s %9.3fs %10" PRId64 " %12.3f\n", scope_len,
                Perfm::LabelName(s.label).c_str(),
                double(s.elapsed_nanos) / 1e9, s.count,
                s.count * 1e9 / s.elapsed_nanos);
  }
}

int64_t APM::AlignCounts(int64_t d) {
  if (d >= counts_.size()) {
    // Realign s.t. the latest points [d-max_window_seconds..d] are at the
    // front.
    size_t offset = d - max_window_seconds_;
    int i = 0;
    for (int j = offset; j < counts_.size(); j++) {
      counts_[i++] = counts_[j];
    }
    for (; i < counts_.size(); i++) {
      counts_[i] = 0;
    }
    t_start_ = t_start_ + std::chrono::seconds(offset);
    return d - offset;
  }
  return d;
}

void APM::Increment(int n) {
  int64_t d = std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::high_resolution_clock::now() - t_start_)
                  .count();
  std::scoped_lock<std::mutex> lk(mut_);
  d = AlignCounts(d);
  counts_[d] += n;
}

double APM::Rate(int window_seconds) {
  if (counts_.empty()) {
    // No data collected yet.
    return 0;
  }
  int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now() - t_start_)
                   .count();
  int64_t quot = us / 1'000'000;
  int64_t rem = us % 1'000'000;

  std::scoped_lock<std::mutex> lk(mut_);
  quot = AlignCounts(quot);
  int64_t w = std::min(static_cast<int64_t>(window_seconds), quot) + 1;
  int64_t sum = std::accumulate(counts_.end() - w, counts_.end(), 0);
  double t =
      static_cast<double>(w) - static_cast<double>(1'000'000 - rem) / 1e6;
  return static_cast<double>(sum) / t;
}

}  // namespace hexz
