#ifndef NDLL_PIPELINE_UTIL_WORKER_THREAD_H_
#define NDLL_PIPELINE_UTIL_WORKER_THREAD_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/util/nvml.h"

namespace ndll {

class WorkerThread {
public:
  typedef std::function<void(void)> Work;
  
  inline WorkerThread(int device_id, bool set_affinity) :
    running_(true), work_complete_(true) {
    nvml::Init();
    thread_ = std::thread(&WorkerThread::ThreadMain,
        this, device_id, set_affinity);
  }

  inline ~WorkerThread() {
    // Wait for work to find errors
    WaitForWork();

    // Mark the thread as not running
    std::unique_lock<std::mutex> lock(mutex_);
    running_ = false;
    cv_.notify_one();
    lock.unlock();

    // Join the thread
    thread_.join();
    nvml::Shutdown();
  }

  inline void DoWork(Work work) {
    std::unique_lock<std::mutex> lock(mutex_);
    work_queue_.push(work);
    work_complete_ = false;
    cv_.notify_one();
  }

  inline void WaitForWork() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!work_complete_) {
      completed_.wait(lock);
    }

    // Check for errors
    if (!errors_.empty()) {
      string error = "Error in worker thread: " +
        errors_.front();
      errors_.pop();
      throw std::runtime_error(error);
    }
  }
  
private:
  void ThreadMain(int device_id, bool set_affinity) {
    CUDA_CALL(cudaSetDevice(device_id));
    if (set_affinity) {
      nvml::SetCPUAffinity();
    }

    while (running_) {
      // Check the queue for work
      std::unique_lock<std::mutex> lock(mutex_);
      while (work_queue_.empty() && running_) {
        cv_.wait(lock);
      }

      if (!running_) break;
      
      Work work = work_queue_.front();
      work_queue_.pop();
      lock.unlock();
      
      try {
        work();
      } catch(std::runtime_error &e) {
        lock.lock();
        errors_.push(e.what());
        lock.unlock();
      } catch(...) {
        lock.lock();
        errors_.push("Caught unknown exception in thread.");
        lock.unlock();
      }

      lock.lock();
      if (work_queue_.empty()) {
        work_complete_ = true;
        completed_.notify_one();
      }
    }
  }

  bool running_, work_complete_;
  std::queue<Work> work_queue_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_, completed_;

  std::queue<string> errors_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_UTIL_WORKER_THREAD_H_
