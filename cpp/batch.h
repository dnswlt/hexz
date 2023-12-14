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
  // Token is a RAII class used for registering and unregistering threads from
  // the Batcher.
  class Token {
   public:
    explicit Token(Batcher* b) : b_{b} {}
    Token(const Token&) = delete;
    Token& operator=(Token&) = delete;
    Token(Token&& other) : b_{other.b_} { other.b_ = nullptr; };
    Token& operator=(Token&& other) {
      b_ = other.b_;
      other.b_ = nullptr;
    }
    ~Token() {
      if (b_ != nullptr) {
        b_->Unregister();
      }
    }

   private:
    Batcher* b_;
  };
  Batcher(std::unique_ptr<ComputeT> comp, int batch_size,
          int64_t timeout_micros)
      : comp_(std::move(comp)),
        max_batch_size_(batch_size),
        timeout_micros_(timeout_micros),
        batch_ready_(false),
        waiting_(0) {}

  Token Register() {
    std::scoped_lock<std::mutex> l(m_);
    registered_threads_++;
    return Token(this);
  }

  result_t ComputeValue(input_t v) {
    std::unique_lock<std::mutex> l(m_);
    ABSL_CHECK(registered_threads_ > 0)
        << "Batcher usage error: called ComputeValue with no threads "
           "registered.";
    cv_enter_.wait(l, [this] { return SlotAvailable(); });

    const size_t batch_idx = input_batch_.size();
    input_batch_.push_back(v);
    waiting_++;

    if (SlotAvailable()) {
      // Wait for next batch to be filled. The first thread entering the batch
      // will set a timeout, all others wait indefinitely (i.e. rely on the
      // first thread's timeout).
      if (batch_idx == 0) {
        if (!cv_ready_.wait_for(
                l, std::chrono::microseconds(timeout_micros_),
                [this] { return batch_ready_ || compute_now_; })) {
          // Timed out before batch was filled entirely. Start computation.
          ABSL_LOG(WARNING) << "Timed out at batch size " << waiting_;
          compute_now_ = true;
        }
      } else {
        cv_ready_.wait(l, [this] { return batch_ready_ || compute_now_; });
      }
      if (compute_now_) {
        // Thread was notified to compute, not b/c the batch is ready.
        compute_now_ = false;
        if (!batch_ready_) {
          return ComputeAllAndNotify(std::move(l), batch_idx);
        }
      }

      ABSL_CHECK(!compute_now_ && batch_ready_)
          << "Must only arrive here when batch is ready.";
      waiting_--;
      result_t result = result_batch_[batch_idx];
      if (AllDone()) {
        // This is the last thread leaving the batch.
        MarkBatchDone();
        l.unlock();
        cv_enter_.notify_all();
      }
      return result;
    }
    // No more slots available; this thread got the last slot in the batch and
    // has to compute values for all waiting threads.
    return ComputeAllAndNotify(std::move(l), batch_idx);
  }

  void UpdateComputeT(std::unique_ptr<ComputeT> comp) {
    std::scoped_lock<std::mutex> l(m_);
    comp_ = std::move(comp);
  }

 private:
  void Unregister() {
    std::scoped_lock<std::mutex> l(m_);
    ABSL_CHECK(registered_threads_ > 0);
    registered_threads_--;
    if (waiting_ > 0 && waiting_ == registered_threads_) {
      // All other registered threads are waiting for results.
      // Don't let them hang waiting for a timeout.
      compute_now_ = true;
      cv_ready_.notify_one();
    }
  }

  result_t ComputeAllAndNotify(std::unique_lock<std::mutex> l,
                               size_t batch_idx) {
    ABSL_CHECK(!batch_ready_);
    result_batch_ = comp_->ComputeAll(std::move(input_batch_));

    batch_ready_ = true;
    result_t result = result_batch_[batch_idx];
    waiting_--;
    if (AllDone()) {
      // This is the last thread leaving the batch. Can only happen here
      // if batch size is 1, because this method is called by the thread
      // computing the values, so it's the first one to see the results.
      ABSL_CHECK(result_batch_.size() == 1);
      MarkBatchDone();
      l.unlock();
      cv_enter_.notify_all();
    } else {
      l.unlock();              // avoid pessimization.
      cv_ready_.notify_all();  // notify others waiting on batch results.
    }
    return result;
  }

  // Resets the internal state of this Batcher and its associated ComputeT.
  inline void MarkBatchDone() {
    ABSL_CHECK(waiting_ == 0);
    input_batch_.clear();
    result_batch_.clear();
    batch_ready_ = false;
  }

  // Returns true if all threads waiting for the batch have picked up their
  // results.
  inline bool AllDone() { return waiting_ == 0; }
  // Returns true if another thread can join the current batch.
  inline bool SlotAvailable() {
    return !batch_ready_ && waiting_ < max_batch_size_ &&
           waiting_ < registered_threads_;
  }

  // The object that performs the actual batch computation.
  // Can be updated even while a batch is being collected using the
  // UpdateComputeT() method.
  std::unique_ptr<ComputeT> comp_;

  const int max_batch_size_;
  const int64_t timeout_micros_;

  // Number of threads that have registered with this Batcher. If this
  // number is less than max_batch_size_, batch computations will start
  // once this many threads are waiting for results, thus avoiding timeouts.
  int registered_threads_ = 0;
  // Number of threads waiting for the results of the current batch.
  // Maintained separately from input_batch_ b/c the latter gets moved for
  // computation.
  int waiting_ = 0;
  std::vector<input_t> input_batch_;
  std::vector<result_t> result_batch_;
  // Marker which is true iff Batcher is expecting all waiting threads
  // to pick up their results.
  bool batch_ready_ = false;
  // Signals to a notified thread that it should run the computation.
  // Usually, the computation is run whenever the batch is full. But
  // threads that unregister should set this flag and notify the others.
  bool compute_now_ = false;

  std::mutex m_;
  std::condition_variable cv_enter_;
  std::condition_variable cv_ready_;
};

#endif  // __HEXZ_BATCH_H__