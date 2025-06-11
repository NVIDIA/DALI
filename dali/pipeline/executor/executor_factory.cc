// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
                     size_t bytes_per_sample_hint, ExecutorFlags flags,
                     QueueSizes prefetch_queue_depth) {
  exec2::Executor2::Config cfg{};
  cfg.async_output = false;
  cfg.set_affinity = Test(flags, ExecutorFlags::SetAffinity);
  cfg.thread_pool_threads = num_thread;
  // TODO(michalz): Expose the thread configuration in the Pipeline (?)
  //                Alternatively, use cooperative parallelism with the CPU thread pool (?)
  cfg.operator_threads = std::min(num_thread, 4);
  if (device_id != CPU_ONLY_DEVICE_ID)
    cfg.device = device_id;
  cfg.max_batch_size = batch_size;
  cfg.cpu_queue_depth = prefetch_queue_depth.cpu_size;
  cfg.gpu_queue_depth = prefetch_queue_depth.gpu_size;
  cfg.queue_policy = exec2::QueueDepthPolicy::Legacy;
  switch (flags & ExecutorFlags::StreamPolicyMask) {
    case ExecutorFlags::StreamPolicyPerOperator:
      cfg.stream_policy = exec2::StreamPolicy::PerOperator;
      break;
    case ExecutorFlags::StreamPolicySingle:
      cfg.stream_policy = exec2::StreamPolicy::Single;
      break;
    case ExecutorFlags::StreamPolicyPerBackend:
    default:
      cfg.stream_policy = exec2::StreamPolicy::PerBackend;
      break;
  }
  switch (flags & ExecutorFlags::ConcurrencyMask) {
    case ExecutorFlags::ConcurrencyNone:
      cfg.concurrency = exec2::OperatorConcurrency::None;
      break;
    case ExecutorFlags::ConcurrencyFull:
      cfg.concurrency = exec2::OperatorConcurrency::Full;
      break;
    case ExecutorFlags::ConcurrencyBackend:
    default:
      cfg.concurrency = exec2::OperatorConcurrency::Backend;
      break;
  }
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
      ExecutorType type, T&&... args) {
  // Go over possible combinations and throw otherwise.
  if (type == ExecutorType::AsyncSeparatedPipelined) {
    return std::make_unique<AsyncSeparatedPipelinedExecutor>(std::forward<T>(args)...);
  } else if (type == ExecutorType::Dynamic) {
    return std::make_unique<exec2::Executor2>(MakeExec2Config(std::forward<T>(args)...));
  } else if (type == ExecutorType::AsyncPipelined) {
    bool force_exec2 = ForceExec2();
    if (force_exec2) {
        std::cerr << "\n!!! Forced use of Executor 2.0 !!!" << std::endl;
      return std::make_unique<exec2::Executor2>(MakeExec2Config(std::forward<T>(args)...));
    } else {
     return std::make_unique<AsyncPipelinedExecutor>(std::forward<T>(args)...);
    }
  } else if (type == ExecutorType::SeparatedPipelined) {
    return std::make_unique<SeparatedPipelinedExecutor>(std::forward<T>(args)...);
  } else if (type == ExecutorType::Pipelined) {
    return std::make_unique<PipelinedExecutor>(std::forward<T>(args)...);
  } else if (type == ExecutorType::Simple) {
    return std::make_unique<SimpleExecutor>(std::forward<T>(args)...);
  }
  std::stringstream error;
  error << std::boolalpha;
  error << "No supported executor selected for pipelined = " << Test(type, ExecutorType::Pipelined)
        << ", separated = " << Test(type, ExecutorType::SeparatedFlag)
        << ", async = " << Test(type, ExecutorType::AsyncFlag)
        << ", dynamic = " << Test(type, ExecutorType::DynamicFlag) << std::endl;
  DALI_FAIL(error.str());
}


std::unique_ptr<ExecutorBase> GetExecutor(ExecutorType type, ExecutorFlags flags,
                                          int batch_size, int num_thread, int device_id,
                                          size_t bytes_per_sample_hint,
                                          QueueSizes prefetch_queue_depth) {
  return GetExecutorImpl(
    type, batch_size, num_thread, device_id, bytes_per_sample_hint, flags, prefetch_queue_depth);
}

}  // namespace dali
