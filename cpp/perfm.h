#ifndef __HEXZ_UTIL_H__
#define __HEXZ_UTIL_H__

#include <cassert>
#include <chrono>
#include <string>

namespace hexz {

class Perfm {
 public:
  enum Label {
    Predict = 0,
    FindLeaf = 1,
    MakeMove = 2,
    PlayGame = 3,
    MaxPuctChild = 4,
    Puct = 5,
    NextMoves = 6,
    NeuralMCTS_Run = 7,
    RandomPlayout = 8,
    PredictBatch = 9,
    // Add new labels here. But don't remove StatsSize and increment its value:
    // It indicates the number of labels we have and defines array sizes below.
    //
    // Also add the new enum to the LabelName method below.
    StatsSize = 10,
  };

  // Helper struct to use RAII for initialization and printing final results.
  struct InitScope {
    InitScope() { Perfm::Init(); }
    ~InitScope() { Perfm::PrintStats(); }
  };

  struct Scope {
    Perfm::Label label;
    std::chrono::high_resolution_clock::time_point started;
    Scope(Perfm::Label label)
        : label{label}, started{std::chrono::high_resolution_clock::now()} {}
    ~Scope() {
      Perfm::stats_[label].count++;
      Perfm::stats_[label].elapsed_nanos +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now() - started)
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
        "Predict",       "FindLeaf",     "MakeMove",  "PlayGame",
        "MaxPuctChild",  "Puct",         "NextMoves", "NeuralMCTS::Run",
        "RandomPlayout", "PredictBatch",
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