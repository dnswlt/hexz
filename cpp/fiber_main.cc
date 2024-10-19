#include <boost/fiber/all.hpp>
#include <iostream>

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
  void pop_n(std::vector<T>& dst, int n) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return !q.empty(); });  // Wait until there's data
    while (n > 0 && !q.empty()) {
      dst.push_back(std::move(q.front()));
      q.pop();
      n--;
    }
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

// Shared data structure for request queue
struct Request {
  int fiber_id;  // ID of the fiber making the request
  boost::fibers::promise<int> result_promise;  // To signal when result is ready
  bool done = false;
};

// GPU pipeline thread processes requests in batches
ConcurrentQueue<Request> request_queue;

void fiber_sub_function(int id) {
  std::cout << "Fiber " << id << " submitting GPU request\n";
  boost::fibers::promise<int> promise;
  boost::fibers::future<int> result = promise.get_future();

  // Add request to the GPU pipeline queue
  request_queue.push(Request{id, std::move(promise)});

  // Wait for the result (fiber will yield here)
  int gpu_result = result.get();
  std::cout << "Fiber " << id << " received GPU result: " << gpu_result
            << " on thread " << std::this_thread::get_id() << "\n";
}

// Function for fibers to run
void fiber_function(int id) {
  for (int i = 0; i < 5; ++i) {  // Each fiber runs 5 iterations
    std::cout << "Fiber " << id << " running iteration " << i << "\n";
    // call sub function to simulate stack depth of real-world scenario.
    fiber_sub_function(id);
  }
}

// Function for GPU pipeline thread
void gpu_pipeline_thread(int batch_size) {
  std::cout << "GPU pipeline running on thread " << std::this_thread::get_id()
            << "\n";
  while (true) {
    std::vector<Request> batch;

    // Collect requests for batching
    while (batch.size() < batch_size) {
      request_queue.pop_n(batch, batch_size - batch.size());
      if (batch.back().done) {
        std::cout << "Received final request (done=true). Exiting.\n";
        return;
      }
    }

    // Simulate processing batch of requests on the GPU
    std::cout << "GPU pipeline processing batch of " << batch.size()
              << " requests...\n";
    std::this_thread::sleep_for(
        std::chrono::milliseconds(100));  // Simulate GPU work

    std::cout << "GPU pipeline returning results to fibers\n";
    for (auto& req : batch) {
      req.result_promise.set_value(42);  // Simulate result
    }
  }
}

int main() {
  const int batch_size = 4;
  const int n_fibers = 8;
  // Create GPU pipeline thread
  std::thread gpu_thread([] { gpu_pipeline_thread(batch_size); });

  // Launch fibers
  std::vector<boost::fibers::fiber> fibers;
  for (int i = 0; i < n_fibers; ++i) {
    fibers.emplace_back(fiber_function, i);
  }

  for (auto& fiber : fibers) {
    fiber.join();
  }

  request_queue.push(Request{.done = true});
  gpu_thread.join();

  std::cout << "Main thread finished.\n";
  return 0;
}