#ifndef NDLL_PIPELINE_THREAD_POOL_H_
#define NDLL_PIPELINE_THREAD_POOL_H_

#include <cstdlib>

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "ndll/common.h"
#include "ndll/util/nvml.h"

namespace ndll {

class ThreadPool {
public:
  // Basic unit of work that our threads do
  typedef std::function<void(int)> Work;
  
  inline ThreadPool(int num_thread, int device_id, bool set_affinity)
    : threads_(num_thread),
      running_(true),
      work_complete_(false),
      active_threads_(0) {
    NDLL_ENFORCE(num_thread > 0, "Thread pool must have non-zero size");
    nvml::Init();
    // Start the threads in the main loop
    for (int i = 0; i < num_thread; ++i) {
      threads_[i] = std::thread(std::bind(&ThreadPool::ThreadMain,
              this, i, device_id, set_affinity));
    }
    tl_errors_.resize(num_thread);
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
    nvml::Shutdown();
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
    
    // Check for errors
    for (size_t i = 0; i < threads_.size(); ++i) {
      if (!tl_errors_[i].empty()) {
        // TODO(tgale): What is the desired behavior when
        // more than one error occurs?
        
        // Throw the first error that occured
        string error = "Error in thread " +
          std::to_string(i) + ": " + tl_errors_[i].front();
        throw std::runtime_error(error);
      }
    }
  }
  
  inline int size() const {
    return threads_.size();
  }
  
  DISABLE_COPY_MOVE_ASSIGN(ThreadPool);
private:
  inline void ThreadMain(int thread_id, int device_id, bool set_affinity) {
    CUDA_CALL(cudaSetDevice(device_id));
    if (set_affinity) {
      // Set device affinity. Lock to ensure we don't hit
      // the thread safety issue in NVML
      std::lock_guard<std::mutex> lock(mutex_);
      nvml::SetCPUAffinity();
    }
    
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

      // If an error occurs, we save it in tl_errors_. When
      // WaitForWork is called, we will check for any errors
      // in the threads and return an error if one occured.
      try {
        work(thread_id);
      } catch(std::runtime_error &e) {
        tl_errors_[thread_id].push(e.what());
      } catch(...) {
        tl_errors_[thread_id].push("Caught unknown exception");
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

  // Stored error strings for each thread
  vector<std::queue<string>> tl_errors_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_THREAD_POOL_H_
