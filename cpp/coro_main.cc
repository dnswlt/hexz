#include <coroutine>
#include <iostream>
#include <optional>
// #include <queue>
// #include <thread>
// #include <mutex>
// #include <condition_variable>

/*
Trying to understand coroutines, following
https://www.chiark.greenend.org.uk/~sgtatham/quasiblog/coroutines-c++20/
*/

struct Event {};

class UserFacing {
 public:
  class promise_type;
  using handle_type = std::coroutine_handle<promise_type>;
  // The type returned by a coroutine function typically has
  // a ::promise_type nested type for promise objects. The promise
  // object defines the methods that define the coroutine's behaviour.
  class promise_type {
   public:
    // get_return_object is called by the coroutine runtime when
    // a coroutine function is called. It defines the object that is
    // returned by the coroutine function. This is typically the only
    // type that users of the coroutine interact with.
    UserFacing get_return_object() {
      // A coroutine_handle is necessary to resume execution.
      // A typical pattern is to store this handle inside the
      // user-facing type, and provide a resume() method in it
      // that delegates to the handle's resume() method (see below).
      auto handle = handle_type::from_promise(*this);
      return UserFacing{handle};
    }
    // By returning suspend_never here, we let the coroutine code
    // run until the first co_await on construction. If we return
    // suspend_always here, the coroutine would not execute any code
    // on construction. This is similar to Python's generators, that
    // need an initial next() call to be "primed" for execution.
    std::suspend_never initial_suspend() { return {}; }

    // await_transform gets called when the coroutine calls `co_await event`,
    // to transform the event into an awaitable.
    // This method is not mandatory in a promise. If it is missing, the
    // event being co_await'ed needs to be an awaitable itself already.
    //
    // Its return type indicates whether the coroutine should be suspended
    // at all. Normally, the co_awaited object will do something useful, and
    // the coroutine would be suspended until the result of that work is
    // available.
    //
    // Here we can also register the coroutine (handle) somewhere
    // to be resumed when appropriate.
    std::suspend_always await_transform(Event) { return {}; }

    // yield_value gets called when the coroutine calls `co_yield value`.
    // To make the yielded value available to the caller of the coroutine,
    // it needs to be stored somewhere. See next_value() below for how
    // it gets accessed in a typical generator use case.
    std::suspend_always yield_value(int value) {
      yielded_value_ = value;
      return {};
    }

    // This coroutine does not return anything, so we define a return_void
    // method. If it did return something, we'd have to define a return_value
    // method.
    void return_void() {}
    void unhandled_exception() { std::terminate(); }
    // final_suspend should always suspend.
    std::suspend_always final_suspend() noexcept { return {}; }

    std::optional<int> yielded_value_;
  };

  // Coroutine objects can be moved.
  UserFacing(UserFacing &&rhs) : handle_(rhs.handle_) { rhs.handle_ = nullptr; }
  UserFacing &operator=(UserFacing &&rhs) {
    if (handle_) {
      handle_.destroy();
    }
    handle_ = rhs.handle_;
    rhs.handle_ = nullptr;
    return *this;
  }
  // Destroy the coroutine's handle, which will also destroy the promise,
  // when the user-facing type goes out of scope.
  ~UserFacing() {
    if (handle_) {
      handle_.destroy();
    }
  }
  // Let users of our coroutine resume it.
  void resume() { handle_.resume(); }

  // Get the next value from this coroutine (treating it as a generator).
  // We null out the result before resuming, then resume, and return what
  // the coro put into our yielded_value_ field.
  // After the final co_yield in the coro, nothing will be put in there,
  // and all subsequent calls will return std::nullopt.
  std::optional<int> next_value() {
    auto &promise = handle_.promise();
    promise.yielded_value_ = std::nullopt;
    if (!handle_.done()) {
      // Only resume if the coro isn't done yet!
      handle_.resume();
    }
    return promise.yielded_value_;
  }

 private:
  UserFacing(handle_type handle) : handle_{handle} {}
  // Coroutine objects are not copyable (to avoid double freeing and other
  // issues).
  UserFacing(const UserFacing &) = delete;
  UserFacing &operator=(const UserFacing &) = delete;

  handle_type handle_;
};

UserFacing demo_awaiting_coroutine() {
  std::cout << "we're about to suspend this coroutine" << std::endl;
  co_await Event{};
  std::cout << "we've successfully resumed this coroutine" << std::endl;
}

UserFacing demo_yielding_coroutine() {
  co_yield 100;
  for (int i = 0; i < 4; i++) {
    co_yield i;
  }
  co_yield -1;
}

int main() {
  {
    UserFacing demo_instance = demo_awaiting_coroutine();
    std::cout << "back in main" << std::endl;
    demo_instance.resume();
    std::cout << "... and back in main again" << std::endl;
  }
  {
    UserFacing gen = demo_yielding_coroutine();
    while (std::optional<int> val = gen.next_value()) {
      std::cout << "Next value yielded: " << *val << std::endl;
    }
  }
}

// struct GPUResult {
//     int data; // Some result data
// };

// struct GPUAwaitable {
//     std::coroutine_handle<> awaitingCoroutine;
//     GPUResult result;

//     bool await_ready() const noexcept { return false; } // Always suspend
//     void await_suspend(std::coroutine_handle<> coroutine) noexcept {
//         awaitingCoroutine = coroutine;
//     }
//     GPUResult await_resume() noexcept { return result; }
// };

// struct WorkerCoroutine {
//     struct promise_type;
//     using handle_type = std::coroutine_handle<promise_type>;

//     struct promise_type {
//         WorkerCoroutine get_return_object() {
//             return WorkerCoroutine{handle_type::from_promise(*this)};
//         }
//         std::suspend_always initial_suspend() { return {}; }
//         std::suspend_always final_suspend() noexcept { return {}; }
//         void return_void() {}
//         void unhandled_exception() { std::terminate(); }
//     };

//     handle_type coro_handle;

//     WorkerCoroutine(handle_type h) : coro_handle(h) {}
//     ~WorkerCoroutine() { if (coro_handle) coro_handle.destroy(); }

//     // Coroutine body - this will compute and wait for the result
//     static WorkerCoroutine run(GPUAwaitable& awaitable) {
//         std::cout << "Worker: Starting work...\n";

//         // Simulate work done by the coroutine
//         co_await std::suspend_always{};  // Simulating doing some work

//         std::cout << "Worker: Submitting to GPU and waiting for result...\n";
//         GPUResult result = co_await awaitable; // Suspend here

//         std::cout << "Worker: Got result from GPU: " << result.data << "\n";
//     }
// };

// std::queue<GPUAwaitable*> gpuQueue;
// std::mutex gpuQueueMutex;
// std::condition_variable gpuQueueCV;

// void gpu_thread_function() {
//     while (true) {
//         std::unique_lock lock(gpuQueueMutex);
//         gpuQueueCV.wait(lock, [] { return !gpuQueue.empty(); });

//         // Process all requests in the queue
//         while (!gpuQueue.empty()) {
//             GPUAwaitable* awaitable = gpuQueue.front();
//             gpuQueue.pop();

//             // Simulate GPU processing
//             std::this_thread::sleep_for(std::chrono::milliseconds(100)); //
//             GPU work awaitable->result = GPUResult{ 42 }; // Simulate a
//             result

//             // Resume the waiting coroutine
//             awaitable->awaitingCoroutine.resume();
//         }
//     }
// }

// void submit_to_gpu(GPUAwaitable& awaitable) {
//     std::lock_guard lock(gpuQueueMutex);
//     gpuQueue.push(&awaitable);
//     gpuQueueCV.notify_one();
// }

// int main() {
//     // Start the GPU processing thread
//     std::thread gpuThread(gpu_thread_function);

//     // Create and run two worker coroutines
//     GPUAwaitable awaitable1, awaitable2;
//     WorkerCoroutine::run(awaitable1);
//     WorkerCoroutine::run(awaitable2);

//     // Submit both requests to the GPU
//     submit_to_gpu(awaitable1);
//     submit_to_gpu(awaitable2);

//     // Wait for the GPU thread to finish (in this example, it won't, so we'll
//     detach it) gpuThread.detach();

//     // Keep the main thread alive while the GPU and coroutines work
//     std::this_thread::sleep_for(std::chrono::seconds(1));

//     std::cout << "Main: Done.\n";
// }
