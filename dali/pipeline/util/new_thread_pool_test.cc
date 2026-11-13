// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/util/new_thread_pool.h"
#include <gtest/gtest.h>
#include <vector>
#include "dali/core/cuda_error.h"
#include "dali/core/dynlink_cuda.h"

namespace dali {
namespace test {

TEST(NewThreadPool, DeviceContextInWorkerThreads) {
  // Regression test: the DeviceGuard set up in OnThreadStart must stay in effect for the whole
  // lifetime of the worker thread. Previously, the guard was (implicitly) copied when the
  // std::any holding it was moved, and the destruction of the source restored the thread's
  // original (null) context before any task could run.
  ASSERT_TRUE(cuInitChecked());
  int ndevs = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndevs));
  ASSERT_GT(ndevs, 0);
  int device_id = ndevs - 1;

  const int num_threads = 4;
  const int num_tasks = 64;
  // Only driver API is used in the tasks - the runtime API could bind a context lazily
  std::vector<CUcontext> task_ctx(num_tasks, nullptr);
  std::vector<CUresult> task_status(num_tasks, CUDA_SUCCESS);
  {
    NewThreadPool tp(num_threads, device_id, false, "DeviceContext test");
    ThreadPoolFacade facade(&tp);
    for (int i = 0; i < num_tasks; i++) {
      facade.AddWork([&, i]() {
        task_status[i] = cuCtxGetCurrent(&task_ctx[i]);
      });
    }
    facade.RunAll();
  }

  CUcontext primary_ctx = nullptr;
  CUDA_CALL(cuDevicePrimaryCtxRetain(&primary_ctx, device_id));
  CUDA_CALL(cuDevicePrimaryCtxRelease(device_id));

  for (int i = 0; i < num_tasks; i++) {
    EXPECT_EQ(task_status[i], CUDA_SUCCESS) << "Task " << i;
    EXPECT_EQ(task_ctx[i], primary_ctx)
        << "Task " << i << " didn't run with the primary context of device " << device_id;
  }
}

}  // namespace test
}  // namespace dali
