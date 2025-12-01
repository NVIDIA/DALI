// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_EXEC_TASKING_EXECUTOR_H_
#define DALI_CORE_EXEC_TASKING_EXECUTOR_H_

#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include "dali/core/exec/tasking/scheduler.h"

namespace dali::tasking {

/** A simple thread-pool-based executor based on the `Scheduler` class.
 *
 * The executor maintains a thread pool which `Pop` and `Run` tasks in a loop.
 */
class Executor : public Scheduler {
 public:
  explicit Executor(int num_threads = std::thread::hardware_concurrency())
      : num_threads_(num_threads) {}

  ~Executor() {
    Shutdown();
  }

  /** Launches the worker threads.
   *
   * Multiple calls to Start have no effect. The function is not thread safe.
   *
   * @param thread_setup A function executed before the main loop.
   */
  void Start(std::function<void()> thread_setup = {}) {
    if (started_)
      return;
    assert(workers_.empty());
    for (int i = 0; i < num_threads_; i++)
      workers_.emplace_back([thread_setup, this]() {
        if (thread_setup)
          thread_setup();
        RunWorker();
      });
    started_ = true;
  }

  /** @brief Notifies the scheduler of the imminent shutdown and waits for the worker threads
   *         to terminate.
   */
  void Shutdown() {
    Scheduler::Shutdown();
    for (auto &w : workers_)
      w.join();
    workers_.clear();
  }

  bool IsRunning() const {
    return started_ && !ShutdownRequested();
  }

 private:
  bool started_ = false;

  void RunWorker() {
    while (SharedTask task = Pop()) {
      task->Run();
    }
  }

  int num_threads_ = -1;
  std::vector<std::thread> workers_;
};

}  // namespace dali::tasking

#endif  // DALI_CORE_EXEC_TASKING_EXECUTOR_H_
