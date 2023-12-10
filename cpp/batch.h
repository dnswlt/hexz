#ifndef __HEXZ_BATCH_H__
#define __HEXZ_BATCH_H__

#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>

#include <condition_variable>
#include <mutex>

template <typename ComputeT>
class Batcher {
 public:
  using input_t = typename ComputeT::input_t;
  using result_t = typename ComputeT::result_t;

  Batcher(ComputeT* comp, int batch_size, int64_t timeout_micros)
      : comp_(comp),
        max_batch_size_(batch_size),
        timeout_micros_(timeout_micros),
        batch_ready_(false),
        done_(0) {}

  result_t ComputeValue(input_t v) {
    std::unique_lock<std::mutex> l(m_);
    cv_enter_.wait(l, [this] { return SlotAvailable(); });

    const size_t batch_idx = input_batch_.size();
    input_batch_.push_back(v);
    input_batch_size_++;

    if (SlotAvailable()) {
      // Wait for next batch to be filled. The first thread entering the batch
      // will set a timeout, all others wait indefinitely (i.e. rely on the
      // first thread's timeout).
      if (batch_idx == 0) {
        if (!cv_ready_.wait_for(l, std::chrono::microseconds(timeout_micros_),
                                [this] { return batch_ready_; })) {
          // Timed out before batch was filled entirely. Start computation.
          ABSL_LOG(WARNING) << "Timed out at batch size " << input_batch_size_;
          return ComputeAllAndNotify(std::move(l), batch_idx);
        }
      } else {
        cv_ready_.wait(l, [this] { return batch_ready_; });
      }
      ABSL_CHECK(batch_ready_) << "Must only arrive here when batch is ready.";

      done_++;
      result_t result = result_batch_[batch_idx];
      if (AllDone()) {
        // This is the last thread leaving the batch.
        BatchDone();
        l.unlock();
        cv_enter_.notify_all();
      }
      return result;
    }
    // No more slots available; this thread got the last slot in the batch and
    // has to compute values for all waiting threads.
    return ComputeAllAndNotify(std::move(l), batch_idx);
  }

  void UpdateComputeT(ComputeT* comp) {
    std::scoped_lock<std::mutex> l(m_);
    comp_ = comp;
  }

 private:
  result_t ComputeAllAndNotify(std::unique_lock<std::mutex> l,
                               size_t batch_idx) {
    result_batch_ = comp_->ComputeAll(std::move(input_batch_));

    batch_ready_ = true;
    result_t result = result_batch_[batch_idx];
    done_++;
    if (AllDone()) {
      // This is the last thread leaving the batch. Can only happen here
      // if batch size is 1, because this method is called by the thread
      // computing the values, so it's the first one to see the results.
      ABSL_CHECK(input_batch_size_ == 1);
      BatchDone();
      l.unlock();
      cv_enter_.notify_all();
    } else {
      l.unlock();              // avoid pessimization.
      cv_ready_.notify_all();  // notify others waiting on batch results.
    }
    return result;
  }

  // Resets the internal state of this Batcher and its associated ComputeT.
  void BatchDone() {
    input_batch_size_ = 0;
    input_batch_.clear();
    result_batch_.clear();
    batch_ready_ = false;
    done_ = 0;
  }

  // Returns true if all threads waiting for the batch have picked up their
  // results.
  inline bool AllDone() { return done_ == input_batch_size_; }
  // Returns true if another thread can join the current batch.
  inline bool SlotAvailable() {
    return !batch_ready_ && max_batch_size_ - input_batch_size_;
  }

  // Pointer to object that performs the actual computation.
  // Not owned! (We use a pointer here so that comp_ can be updated, even
  // in the middle of a batch operation.)
  ComputeT* comp_;

  const int max_batch_size_;
  const int64_t timeout_micros_;

  // Maintained separately from input_batch_ b/c the latter gets moved for
  // computation.
  int input_batch_size_ = 0;
  std::vector<input_t> input_batch_;
  std::vector<result_t> result_batch_;
  // Marker which is true iff Batcher is expecting all waiting threads
  // to pick up their results.
  bool batch_ready_ = false;
  // Number of tasks that have retrieved their value from the current batch
  // result.
  int done_ = 0;
  std::mutex m_;
  std::condition_variable cv_enter_;
  std::condition_variable cv_ready_;
};

#endif  // __HEXZ_BATCH_H__