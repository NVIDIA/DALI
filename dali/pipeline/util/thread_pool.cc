// Copyright (c) 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <cstdlib>
#include <utility>
#include "dali/pipeline/util/thread_pool.h"
#include "dali/test/timing.h"
#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif
#include "dali/core/format.h"
#include "dali/core/cuda_error.h"
#include "dali/core/device_guard.h"
#include "dali/core/nvtx.h"

namespace dali {

using perf_timer_t = std::chrono::high_resolution_clock;

namespace detail {
struct alignas(64) LockStats {
  LockStats() = default;
  explicit LockStats(bool print) : print(print) {}

  ~LockStats() {
    if (print) {
      std::cout << "ThreadPool mutex timing statistics:";
      if (queue_pop_locks) {
        std::cout << "\nPop:                ";
        test::print_time(std::cout, 1e-9 * queue_pop_lock_time/queue_pop_locks);
        std::cout << " (max: ";
        test::print_time(std::cout, 1e-9 * queue_pop_lock_time_max);
        std::cout << ", min: ";
        test::print_time(std::cout, 1e-9 * queue_pop_lock_time_min);
        std::cout << ", events: " << queue_pop_locks << ")\n";
      }
      if (queue_push_locks) {
        std::cout << "\nPush:                ";
        test::print_time(std::cout, 1e-9 * queue_push_lock_time/queue_push_locks);
        std::cout << " (max: ";
        test::print_time(std::cout, 1e-9 * queue_push_lock_time_max);
        std::cout << ", min: ";
        test::print_time(std::cout, 1e-9 * queue_push_lock_time_min);
        std::cout << ", events: " << queue_push_locks << ")\n";
      }
      if (completed_notify_locks) {
        std::cout << "\nCompleted (notify): ";
        test::print_time(std::cout, 1e-9 * completed_notify_lock_time/completed_notify_locks);
        std::cout << " (max: ";
        test::print_time(std::cout, 1e-9 * completed_notify_lock_time_max);
        std::cout << ", min: ";
        test::print_time(std::cout, 1e-9 * completed_notify_lock_time_min);
        std::cout << ", events: " << completed_notify_locks << ")\n";
      }
      if (completed_wait_locks) {
        std::cout << "\nCompleted (wait):   ";
        test::print_time(std::cout, 1e-9 * completed_wait_lock_time/completed_wait_locks);
        std::cout << " (max: ";
        test::print_time(std::cout, 1e-9 * completed_wait_lock_time_max);
        std::cout << ", min: ";
        test::print_time(std::cout, 1e-9 * completed_wait_lock_time_min);
        std::cout << ", events: " << completed_wait_locks << ")\n";
      }
      std::cout << std::endl;
    }
  }

  bool print = false;

  LockStats &operator+=(const LockStats &x) {
    queue_pop_lock_time += x.queue_pop_lock_time;
    queue_pop_lock_time_max.store(std::max(queue_pop_lock_time_max.load(), x.queue_pop_lock_time_max.load()));
    queue_pop_lock_time_min.store(std::min(queue_pop_lock_time_min.load(), x.queue_pop_lock_time_min.load()));
    queue_push_lock_time += x.queue_push_lock_time;
    queue_push_lock_time_max.store(std::max(queue_push_lock_time_max.load(), x.queue_push_lock_time_max.load()));
    queue_push_lock_time_min.store(std::min(queue_push_lock_time_min.load(), x.queue_push_lock_time_min.load()));
    completed_notify_lock_time += x.completed_notify_lock_time;
    completed_notify_lock_time_max.store(std::max(completed_notify_lock_time_max.load(), x.completed_notify_lock_time_max.load()));
    completed_notify_lock_time_min.store(std::min(completed_notify_lock_time_min.load(), x.completed_notify_lock_time_min.load()));
    completed_wait_lock_time += x.completed_wait_lock_time;
    queue_push_locks += x.queue_push_locks;
    queue_pop_locks += x.queue_pop_locks;
    completed_notify_locks += x.completed_notify_locks;
    completed_wait_locks += x.completed_wait_locks;
    return *this;
  }

  std::atomic<int64_t>  queue_pop_lock_time{0},
                        queue_pop_lock_time_max{0},
                        queue_pop_lock_time_min{0},
                        queue_push_lock_time{0},
                        queue_push_lock_time_max{0},
                        queue_push_lock_time_min{0},
                        completed_notify_lock_time{0},
                        completed_notify_lock_time_max{0},
                        completed_notify_lock_time_min{0},
                        completed_wait_lock_time{0},
                        completed_wait_lock_time_max{0},
                        completed_wait_lock_time_min{0};
  std::atomic<int64_t>  queue_pop_locks{0},
                        queue_push_locks{0},
                        completed_notify_locks{0},
                        completed_wait_locks{0};
};

LockStats g_stats(true);
}  // namespace detail

ThreadPool::ThreadPool(int num_thread, int device_id, bool set_affinity, const char* name)
    : threads_(num_thread), running_(true), started_(false), outstanding_work_(0) {
  DALI_ENFORCE(num_thread > 0, "Thread pool must have non-zero size");
  stats_ = std::make_unique<detail::LockStats>(true);
  if (name)
      name_ = name;
#if NVML_ENABLED
  // We use NVML only for setting thread affinity
  if (device_id != CPU_ONLY_DEVICE_ID && set_affinity) {
    nvml_handle_ = nvml::NvmlInstance::CreateNvmlInstance();
  }
#endif
  // Start the threads in the main loop
  for (int i = 0; i < num_thread; ++i) {
    threads_[i] = std::thread(std::bind(&ThreadPool::ThreadMain, this, i, device_id, set_affinity,
                                        make_string("[DALI][TP", i, "]", name)));
  }
  tl_errors_.resize(num_thread);
}

ThreadPool::~ThreadPool() {
  WaitForWork(false);

  std::unique_lock<std::mutex> lock(mutex_);
  running_ = false;
  condition_.notify_all();
  lock.unlock();

  for (auto &thread : threads_) {
    thread.join();
  }
  std::cout << "Shutting down thread pool " << name_ << "\n";
  detail::g_stats += *stats_;
}

void ThreadPool::AddWork(Work work, int64_t priority, bool start_immediately) {
  bool started_before = started_;
  outstanding_work_.fetch_add(1);
  if (started_before) {
    auto lock_start = perf_timer_t::now();
    std::lock_guard<std::mutex> lock(mutex_);
    auto lock_time = (perf_timer_t::now() - lock_start).count();
    stats_->queue_push_lock_time += lock_time;
    stats_->queue_push_lock_time_max.store(std::max(stats_->queue_push_lock_time_max.load(), lock_time));
    stats_->queue_push_lock_time_min.store(std::min(stats_->queue_push_lock_time_min.load(), lock_time));
    stats_->queue_push_locks++;
    work_queue_.push({priority, std::move(work)});
  } else {
    work_queue_.push({priority, std::move(work)});
    if (start_immediately) {
      auto lock_start = perf_timer_t::now();
      std::lock_guard<std::mutex> lock(mutex_);
      auto lock_time = (perf_timer_t::now() - lock_start).count();
      stats_->queue_push_lock_time += lock_time;
      stats_->queue_push_lock_time_max.store(std::max(stats_->queue_push_lock_time_max.load(), lock_time));
      stats_->queue_push_lock_time_min.store(std::min(stats_->queue_push_lock_time_min.load(), lock_time));
      stats_->queue_push_locks++;
      started_ = true;
    }
  }
  if (started_) {
    if (started_before)
      condition_.notify_one();
    else
      condition_.notify_all();
  }
}

// Blocks until all work issued to the thread pool is complete
void ThreadPool::WaitForWork(bool checkForErrors) {
  if (outstanding_work_.load()) {
    auto lock_start = perf_timer_t::now();
    std::unique_lock<std::mutex> lock(completed_mutex_);
    completed_.wait(lock, [&, this] {
      bool ready = this->outstanding_work_ == 0;
      if (ready)
        lock_start = perf_timer_t::now();
      return ready;
    });
    auto lock_time = (perf_timer_t::now() - lock_start).count();
    stats_->completed_wait_lock_time += lock_time;
    stats_->completed_wait_lock_time_max.store(std::max(stats_->completed_wait_lock_time_max.load(), lock_time));
    stats_->completed_wait_lock_time_min.store(std::min(stats_->completed_wait_lock_time_min.load(), lock_time));
    stats_->completed_wait_locks++;
  }
  started_ = false;
  if (checkForErrors) {
    // Check for errors
    for (size_t i = 0; i < threads_.size(); ++i) {
      if (!tl_errors_[i].empty()) {
        // Throw the first error that occurred
        string error = make_string("Error in thread ", i, ": ", tl_errors_[i].front());
        tl_errors_[i].pop();
        throw std::runtime_error(error);
      }
    }
  }
}

void ThreadPool::RunAll(bool wait) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = true;
  }
  condition_.notify_all();  // other threads will be waken up if needed
  if (wait) {
    WaitForWork();
  }
}

int ThreadPool::NumThreads() const {
  return threads_.size();
}

std::vector<std::thread::id> ThreadPool::GetThreadIds() const {
  std::vector<std::thread::id> tids;
  tids.reserve(threads_.size());
  for (const auto &thread : threads_)
    tids.emplace_back(thread.get_id());
  return tids;
}


void ThreadPool::ThreadMain(int thread_id, int device_id, bool set_affinity,
                            const std::string &name) {
  SetThreadName(name.c_str());
  DeviceGuard g(device_id);
  try {
#if NVML_ENABLED
    if (set_affinity) {
      const char *env_affinity = std::getenv("DALI_AFFINITY_MASK");
      int core = -1;
      if (env_affinity) {
        const auto &vec = string_split(env_affinity, ',');
        if ((size_t)thread_id < vec.size()) {
          core = std::stoi(vec[thread_id]);
        } else {
          DALI_WARN("DALI_AFFINITY_MASK environment variable is set, "
                    "but does not have enough entries: thread_id (", thread_id,
                    ") vs #entries (", vec.size(), "). Ignoring...");
        }
      }
      nvml::SetCPUAffinity(core);
    }
#endif
  } catch (std::exception &e) {
    tl_errors_[thread_id].push(e.what());
  } catch (...) {
    tl_errors_[thread_id].push("Caught unknown exception");
  }

  detail::LockStats thread_stats;

  while (running_) {
#if 0
    // We have noticed that for large number of threads on aarch64, the threads trying to
    // lock the mutex are waiting for a long time.
    // This snippet tries to alleviate this issue by trying to lock the mutex
    // with a try_to_lock. If the mutex is not locked, the thread will sleep for
    // a short time and try again.
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      for (int wait = 1;; wait = std::max(wait * 2, 16)) {
        std::this_thread::sleep_for(std::chrono::microseconds(wait));
        if (lock.try_lock())
          break;
      }
    }
#else
    auto lock_start = perf_timer_t::now();
    std::unique_lock<std::mutex> lock(mutex_);
    auto lock_time = (perf_timer_t::now() - lock_start).count();
    thread_stats.queue_pop_lock_time += lock_time;
    thread_stats.queue_pop_lock_time_max.store(std::max(thread_stats.queue_pop_lock_time_max.load(), lock_time));
    thread_stats.queue_pop_lock_time_min.store(std::min(thread_stats.queue_pop_lock_time_min.load(), lock_time));
    thread_stats.queue_pop_locks++;
#endif

    condition_.wait(lock, [&, this] {
      bool ret = !running_ || (!work_queue_.empty() && started_);
      if (ret)
        lock_start = perf_timer_t::now();
      return ret;
    });
    thread_stats.queue_pop_lock_time += (perf_timer_t::now() - lock_start).count();
    thread_stats.queue_pop_locks++;

    // If we're no longer running, exit the run loop
    if (!running_) break;

    // Get work from the queue.
    Work work = std::move(work_queue_.top().second);
    work_queue_.pop();

    // Unlock the lock
    lock.unlock();

    // If an error occurs, we save it in tl_errors_. When
    // WaitForWork is called, we will check for any errors
    // in the threads and return an error if one occured.
    try {
      work(thread_id);
    } catch (std::exception &e) {
      lock.lock();
      tl_errors_[thread_id].push(e.what());
      lock.unlock();
    } catch (...) {
      lock.lock();
      tl_errors_[thread_id].push("Caught unknown exception");
      lock.unlock();
    }

    // The task is now complete - we can atomically decrement the number of outstanding work.
    // If it reaches zero, we must safely notify the potential threads waiting for the work
    // to complete.
    // NOTE: We don't have to acquire the mutex until the number of waiting threads reaches 0.
    if (--outstanding_work_ == 0) {
      // We don't need to guard the modification of the atomic value with a mutex -
      // however, we need to lock it briefly to make sure we don't have this scenario:
      //
      // worker                           WaitForWork
      //
      //                                  lock.lock()
      //                                  return outstanding_work_ == 0  (false!)
      // --outstanding_work == 0 (true)
      // compleded_.notify_all()          NOT WAITING FOR compleded_ YET!!!!!!!!!!!!!
      //                                  atomically unlock `lock` and wait for `completed_`
      //                                                               ^^^^ deadlock


      // The brief lock/unlock sequence avoids the above.
      // The call to lock.lock() prevents the worker thread from signalling the event while
      // the control thread is evaluating the condition (which happens with the mutex owned).
      // Now it looks like this:
      //
      // worker                           WaitForWork
      //
      //                                  lock.lock()
      //                                  return outstanding_work_ == 0  (false!)
      // --outstanding_work == 0 (true)
      // lock.lock(
      //                                  atomically unlock `lock` and wait for `completed_`
      // At this point we know that if
      // anyone was executing WaitForWork
      // they're not evaluating the
      // condition but rather waiting on
      // the completed_ condvar.
      //
      // lock.unlock()
      // compleded_.notify_all()
      //                                  notified - wake up
      //                                  lock.lock()
      //                                  continue execution
      {
        lock_start = perf_timer_t::now();
        std::lock_guard<std::mutex> lock2(completed_mutex_);
        auto lock_time = (perf_timer_t::now() - lock_start).count();
        thread_stats.completed_notify_lock_time += lock_time;
        thread_stats.completed_notify_lock_time_max.store(std::max(thread_stats.completed_notify_lock_time_max.load(), lock_time));
        thread_stats.completed_notify_lock_time_min.store(std::min(thread_stats.completed_notify_lock_time_min.load(), lock_time));
        thread_stats.completed_notify_locks++;
      }
      completed_.notify_all();
    }
  }
  *stats_ += thread_stats;
}

}  // namespace dali
