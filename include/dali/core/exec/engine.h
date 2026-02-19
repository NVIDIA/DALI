// Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_EXEC_ENGINE_H_
#define DALI_CORE_EXEC_ENGINE_H_

#include <utility>

namespace dali {

/*
concept ExecutionEngine {
  /// @brief Adds work to the engine
  /// @param f           work item, callable with `int thread_idx`
  /// @param priority    priority hint for the job, the higher, the earlier it should start
  /// @param start_immediately        if true, all jobs can start - it's just a hint
  ///                                 and implementations may start running the jobs earlier
  void AddWork(CallableWithInt f, int64_t priority, bool start_immediately = false);

  /// @brief Starts the work and waits for it to complete.
  /// If there was an exception in one of the jobs, rethrows one of them.
  void RunAll();

  /// @brief Returns number of threads in this execution engine.
  int NumThreads() const noexcept;
};
*/

/**
 * @brief Implements a fake thread-pool-like object which immediately executes any work submitted
 */
class SequentialExecutionEngine {
 public:
  /**
   * @brief Immediately execute a callable object `f` with thread index 0.
   */
  template <typename FunctionLike>
  void AddWork(FunctionLike &&f, int64_t priority = 0) {
    const int idx = 0;  // use of 0 literal would successfully call f expecting a pointer
    f(idx);
  }

  void RunAll() {}

  /**
   * @brief Returns 1
   */
  constexpr int NumThreads() const noexcept { return 1; }
};

}  // namespace dali

#endif  // DALI_CORE_EXEC_ENGINE_H_
