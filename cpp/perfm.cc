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

}  // namespace hexz
