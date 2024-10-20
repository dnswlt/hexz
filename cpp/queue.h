#pragma once

#include <mutex>
#include <queue>
#include <vector>

namespace hexz {

template <typename T>
class BoundedConcurrentQueue {
 public:
  // Constructor: Initializes the queue with a given capacity
  explicit BoundedConcurrentQueue(size_t capacity) : max_capacity(capacity) {}

  // Pushes an item to the queue. Blocks if the queue is full.
  void push(const T& item) {
    std::unique_lock<std::mutex> lock(mtx);
    cv_not_full.wait(lock, [this] {
      return q.size() < max_capacity;
    });  // Wait if the queue is full
    q.push(item);
    cv_not_empty.notify_one();  // Notify consumers that an item is available
  }

  // Pushes an item to the queue using move semantics. Blocks if the queue is
  // full.
  void push(T&& item) {
    std::unique_lock<std::mutex> lock(mtx);
    cv_not_full.wait(lock, [this] {
      return q.size() < max_capacity;
    });  // Wait if the queue is full
    q.push(std::move(item));
    cv_not_empty.notify_one();  // Notify consumers that an item is available
  }

  // Pops an item from the queue. Blocks if the queue is empty.
  T pop() {
    std::unique_lock<std::mutex> lock(mtx);
    cv_not_empty.wait(
        lock, [this] { return !q.empty(); });  // Wait if the queue is empty
    T item = std::move(q.front());
    q.pop();
    cv_not_full.notify_one();  // Notify producers that space is available
    return item;
  }

  // Tries to pop an item without blocking. Returns false if the queue is empty.
  bool try_pop(T& item) {
    std::lock_guard<std::mutex> lock(mtx);
    if (q.empty()) {
      return false;
    }
    item = std::move(q.front());
    q.pop();
    cv_not_full.notify_one();  // Notify producers that space is available
    return true;
  }

  // Checks if the queue is empty
  bool empty() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.empty();
  }

  // Checks if the queue is full
  bool full() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.size() >= max_capacity;
  }

  // Returns the size of the queue
  size_t size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.size();
  }

 private:
  std::queue<T> q;
  size_t max_capacity;     // Maximum number of elements the queue can hold
  mutable std::mutex mtx;  // Mutex for synchronizing access
  std::condition_variable
      cv_not_full;  // Condition variable for "not full" state
  std::condition_variable
      cv_not_empty;  // Condition variable for "not empty" state
};

template <typename T>
class ConcurrentQueue {
 public:
  // Adds an element to the queue
  void push(const T& item) {
    std::lock_guard<std::mutex> lock(mtx);
    q.push(item);
    cv.notify_one();  // Notify one waiting thread
  }

  // Adds an element using move semantics
  void push(T&& item) {
    std::lock_guard<std::mutex> lock(mtx);
    q.push(std::move(item));
    cv.notify_one();  // Notify one waiting thread
  }

  // Pops an element from the queue
  // Blocks if the queue is empty
  T pop() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return !q.empty(); });  // Wait until there's data
    T item = std::move(q.front());
    q.pop();
    return item;
  }

  // Moves up to n items from the queue into the given vector dst.
  // Blocks if the queue is empty.
  // Returns the number of items added to dst.
  int pop_n(std::vector<T>& dst, int n) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return !q.empty(); });  // Wait until there's data
    int k;
    for (k = 0; k < n && !q.empty(); k++) {
      dst.push_back(std::move(q.front()));
      q.pop();
    }
    return k;
  }

  // Tries to pop an element without blocking
  bool try_pop(T& item) {
    std::lock_guard<std::mutex> lock(mtx);
    if (q.empty()) {
      return false;
    }
    item = std::move(q.front());
    q.pop();
    return true;
  }

  // Check if the queue is empty
  bool empty() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.empty();
  }

  // Get the size of the queue
  size_t size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.size();
  }

 private:
  std::queue<T> q;
  mutable std::mutex mtx;      // Mutex for synchronizing access
  std::condition_variable cv;  // Condition variable to signal waiting threads
};

}  // namespace hexz
