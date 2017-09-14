#ifndef NDLL_PIPELINE_THREAD_POOL_H_
#define NDLL_PIPELINE_THREAD_POOL_H_

#include <cstdlib>

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "ndll/common.h"

namespace ndll {

class ThreadPool {
public:
  // Basic unit of work that our threads do
  typedef std::function<void(int)> Work;
  
  inline ThreadPool(int num_thread)
    : threads_(num_thread),
      running_(true),
      work_complete_(false),
      active_threads_(0) {
    NDLL_ENFORCE(num_thread > 0, "Thread pool must have non-zero size");
    // Start the threads in the main loop
    for (int i = 0; i < num_thread; ++i) {
      threads_[i] = std::thread(std::bind(&ThreadPool::ThreadMain, this, i));
    }
  }

  inline ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
    }
    condition_.notify_all();

    for (auto &thread : threads_) {
      thread.join();
    }
  }

  inline void DoWorkWithID(Work work) {
    {
      // Add work to the queue
      std::lock_guard<std::mutex> lock(mutex_);
      work_queue_.push(work);
      work_complete_ = false;
    }
    // Signal a thread to complete the work
    condition_.notify_one();
  }

  // Blocks until all work issued to the thread pool is complete
  inline void WaitForWork() {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_.wait(lock, [this] { return this->work_complete_; });
  }
  
  inline int size() const {
    return threads_.size();
  }
  
  DISABLE_COPY_MOVE_ASSIGN(ThreadPool);
private:
  inline void ThreadMain(int thread_id) {
    while (running_) {
      // Block on the condition to wait for work
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this] { return !(running_ && work_queue_.empty()); });
      
      // If we're no longer running, exit the run loop
      if (!running_) break;

      // Get work from the queue & mark
      // this thread as active
      Work work = work_queue_.front();
      work_queue_.pop();
      ++active_threads_;
      
      // Unlock the lock
      lock.unlock();

      // TODO(tgale): Send the errors back to the main thread
      try {
        work(thread_id);
      } catch(NDLLException &e) {
        cout << "Caught exception in thread " << e.what() << endl;
        exit(EXIT_FAILURE);
      }

      // Mark this thread as idle & check for complete work
      lock.lock();
      --active_threads_;
      if (work_queue_.empty() && active_threads_ == 0) {
        work_complete_ = true;
        completed_.notify_one();
      }
    }
  }
  vector<std::thread> threads_;
  std::queue<Work> work_queue_;
  
  bool running_;
  bool work_complete_;
  int active_threads_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable completed_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_THREAD_POOL_H_
