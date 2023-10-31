#ifndef __HEXZ_UTIL_H__
#define __HEXZ_UTIL_H__

#include <cassert>
#include <chrono>
#include <string>

namespace hexz {

std::string GetEnv(const std::string& name);
int GetEnvAsInt(const std::string& name, int default_value);

int64_t UnixMicros();

class Perfm {
 public:
  enum Label {
    Predict = 0,
    FindLeaf = 1,
    MakeMove = 2,
    // Add new labels here. But don't remove StatsSize and increment its value:
    // It indicates the number of labels we have and defines array sizes below.
    //
    // Also add the new enum to the LabelName method below.
    PlayGameLocally = 3,
    MaxPuctChild = 4,
    Puct = 5,
    PuctMoveProbs = 6,
    StatsSize = 7,
  };

  struct Scope {
    Perfm::Label label;
    std::chrono::steady_clock::time_point started;
    Scope(Perfm::Label label)
        : label{label}, started{std::chrono::steady_clock::now()} {}
    ~Scope() {
      Perfm::stats_[label].count++;
      Perfm::stats_[label].elapsed_nanos +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started)
              .count();
    }
  };

  struct CumulativeStats {
    Perfm::Label label;
    int64_t count = 0;
    int64_t elapsed_nanos;
  };

  static const std::string& LabelName(Perfm::Label label) {
    static std::string names[StatsSize] = {
        "Predict",
        "FindLeaf",
        "MakeMove",
        "PlayGameLocally",
        "MaxPuctChild",
        "Puct",
        "PuctMoveProbs",
    };
    assert(label < StatsSize);
    return names[label];
  }

  static void Init();

  static void PrintStats();

 private:
  static CumulativeStats stats_[StatsSize];
};

}  // namespace hexz
#endif  // __HEXZ_UTIL_H__