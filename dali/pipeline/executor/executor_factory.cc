// Copyright (c) 2019, 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <sstream>
#include <utility>

#include "dali/pipeline/executor/executor_factory.h"
#include "dali/pipeline/executor/pipelined_executor.h"
#include "dali/pipeline/executor/async_pipelined_executor.h"
#include "dali/pipeline/executor/async_separated_pipelined_executor.h"
#include "dali/pipeline/executor/executor2/exec2.h"

namespace dali {

namespace {

auto MakeExec2Config(int batch_size, int num_thread, int device_id,
                     size_t bytes_per_sample_hint, bool set_affinity,
                     QueueSizes prefetch_queue_depth) {
  exec2::Executor2::Config cfg{};
  cfg.async_output = false;
  cfg.set_affinity = set_affinity;
  cfg.thread_pool_threads = num_thread;
  cfg.operator_threads = num_thread;
  if (device_id != CPU_ONLY_DEVICE_ID)
    cfg.device = device_id;
  cfg.max_batch_size = batch_size;
  cfg.cpu_queue_depth = prefetch_queue_depth.cpu_size;
  cfg.gpu_queue_depth = prefetch_queue_depth.gpu_size;
  cfg.queue_policy = exec2::QueueDepthPolicy::Legacy;
  cfg.stream_policy = exec2::StreamPolicy::PerBackend;
  cfg.concurrency = exec2::OperatorConcurrency::Backend;
  return cfg;
}

bool ForceExec2() {
  static bool force_exec2 = []() {
    const char *env = getenv("DALI_USE_EXEC2");
    return env && atoi(env);
  }();
  return force_exec2;
}

}  // namespace

template <typename... T>
std::unique_ptr<ExecutorBase> GetExecutorImpl(
      bool pipelined, bool separated, bool async, bool dynamic, T&&... args) {
  // Go over possible combinations and throw otherwise.
  if (async && separated && pipelined && !dynamic) {
    return std::make_unique<AsyncSeparatedPipelinedExecutor>(std::forward<T>(args)...);
  } else if (async && !separated && pipelined) {
    bool force_exec2 = ForceExec2();
    if (dynamic || force_exec2) {
      if (force_exec2)
        std::cerr << "\n!!! Forced use of Executor 2.0 !!!" << std::endl;
      return std::make_unique<exec2::Executor2>(MakeExec2Config(std::forward<T>(args)...));
    } else {
     return std::make_unique<AsyncPipelinedExecutor>(std::forward<T>(args)...);
    }
  } else if (!async && separated && pipelined && !dynamic) {
    return std::make_unique<SeparatedPipelinedExecutor>(std::forward<T>(args)...);
  } else if (!async && !separated && pipelined && !dynamic) {
    return std::make_unique<PipelinedExecutor>(std::forward<T>(args)...);
  } else if (!async && !separated && !pipelined && !dynamic) {
    return std::make_unique<SimpleExecutor>(std::forward<T>(args)...);
  }
  std::stringstream error;
  error << std::boolalpha;
  error << "No supported executor selected for pipelined = " << pipelined
        << ", separated = " << separated << ", async = " << async
        << ", dynamic = " << dynamic << std::endl;
  DALI_FAIL(error.str());
}


std::unique_ptr<ExecutorBase> GetExecutor(bool pipelined, bool separated, bool async, bool dynamic,
                                          int batch_size, int num_thread, int device_id,
                                          size_t bytes_per_sample_hint, bool set_affinity,
                                          QueueSizes prefetch_queue_depth) {
  return GetExecutorImpl(
    pipelined, separated, async, dynamic,
    batch_size, num_thread, device_id, bytes_per_sample_hint, set_affinity,
    prefetch_queue_depth);
}

}  // namespace dali
