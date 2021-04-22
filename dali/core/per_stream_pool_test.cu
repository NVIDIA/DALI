// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/per_stream_pool.h"  // NOLINT
#include <gtest/gtest.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_error.h"

namespace dali {

void wait_func(void *pvflag) {
  auto *flag = reinterpret_cast<std::atomic_flag*>(pvflag);
  while (flag->test_and_set()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

TEST(PerStreamPool, SingleStream) {
  std::atomic_flag flag;
  flag.test_and_set();

  CUDAStream s1 = CUDAStream::Create(true);
  CUDAStream ssync = CUDAStream::Create(true);
  CUDAEvent e = CUDAEvent::Create();
  PerStreamPool<int> pool;
  int *p1 = nullptr, *p2 = nullptr;
  if (auto lease = pool.Get(s1)) {
    p1 = lease;
  }

  CUDA_CALL(cudaLaunchHostFunc(ssync, wait_func, &flag));
  CUDA_CALL(cudaEventRecord(e, ssync));  // this event is recorded, but not reached, because this
                                         // stream is waiting for a spinning host function

  CUDA_CALL(cudaStreamSynchronize(s1));  // make sure that stream has completed its job
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_EQ(p2, p1) << "Expected to get the same object.";
    CUDA_CALL(cudaStreamWaitEvent(s1, e, 0));  // block s1
  }
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_EQ(p2, p1) << "Expected to get the same object, even if job is still pending.";
  }
  flag.clear();  // unblock stream ssync
  CUDA_CALL(cudaStreamSynchronize(ssync));
  CUDA_CALL(cudaStreamSynchronize(s1));
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_EQ(p2, p1) << "Expected to get the same object.";
  }
}

TEST(PerDevicePool, SingleStreamNoReuse) {
  std::atomic_flag flag;
  flag.test_and_set();

  CUDAStream s1 = CUDAStream::Create(true);
  CUDAStream ssync = CUDAStream::Create(true);
  CUDAEvent e = CUDAEvent::Create();
  PerDevicePool<int> pool;
  int *p1 = nullptr, *p2 = nullptr, *p3 = nullptr;
  if (auto lease = pool.Get(s1)) {
    p1 = lease;
  }
  CUDA_CALL(cudaLaunchHostFunc(ssync, wait_func, &flag));
  CUDA_CALL(cudaEventRecord(e, ssync));  // this event is recorded, but not reached, because this
                                         // stream is waiting for a spinning host function

  CUDA_CALL(cudaStreamSynchronize(s1));  // make sure that stream has completed its job
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_EQ(p2, p1) << "Expected to get the same object.";
    CUDA_CALL(cudaStreamWaitEvent(s1, e, 0));  // block s1
  }
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_NE(p2, p1) << "Expected to get a new object - job is still pending and reuse disabled.";
  }
  flag.clear();  // unblock stream ssync
  CUDA_CALL(cudaStreamSynchronize(ssync));
  CUDA_CALL(cudaStreamSynchronize(s1));
  if (auto lease = pool.Get(s1)) {
    p3 = lease;
    EXPECT_TRUE(p3 == p1 || p3 == p2) << "Expected to get one of the previous objects.";
  }
}

TEST(PerStreamPool, MultiStream) {
  std::atomic_flag flag;
  flag.test_and_set();

  CUDAStream s1 = CUDAStream::Create(true);
  CUDAStream s2 = CUDAStream::Create(true);
  CUDAStream ssync = CUDAStream::Create(true);
  CUDAEvent e = CUDAEvent::Create();
  PerStreamPool<int> pool;
  int *p1 = nullptr, *p2 = nullptr, *p3 = nullptr, *p4 = nullptr;
  if (auto lease = pool.Get(s1)) {
    p1 = lease;
  }

  CUDA_CALL(cudaLaunchHostFunc(ssync, wait_func, &flag));
  CUDA_CALL(cudaEventRecord(e, ssync));  // this event is recorded, but not reached, because this
                                         // stream is waiting for a spinning host function

  CUDA_CALL(cudaStreamSynchronize(s1));  // make sure that stream has completed its job
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_EQ(p2, p1) << "Expected to get the same object.";
    CUDA_CALL(cudaStreamWaitEvent(s1, e, 0));  // block s1
  }
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_EQ(p2, p1) << "Expected to get the same object, even if job is still pending.";
  }
  if (auto lease = pool.Get(s2)) {
    p3 = lease;
    EXPECT_NE(p3, p1) << "Expected to get a new object, job on s1 is still pending.";
  }
  flag.clear();  // unblock stream ssync
  CUDA_CALL(cudaStreamSynchronize(ssync));
  CUDA_CALL(cudaStreamSynchronize(s1));
  CUDA_CALL(cudaStreamSynchronize(s2));
  if (auto lease = pool.Get(s1)) {
    p3 = lease;
    EXPECT_TRUE(p3 == p1 || p3 == p2) << "Expected to get one of the previously objects.";
  }

  CUDA_CALL(cudaLaunchHostFunc(ssync, wait_func, &flag));
  CUDA_CALL(cudaEventRecord(e, ssync));  // this event is recorded, but not reached, because this
                                         // stream is waiting for a spinning host function

  if (auto lease = pool.Get(s1)) {
    p4 = lease;
    EXPECT_TRUE(p4 == p3) << "Expected to get one of the previously objects.";
    CUDA_CALL(cudaStreamWaitEvent(s1, e, 0));  // block s1
  }
  if (auto lease = pool.Get(s2)) {
    p4 = lease;
    EXPECT_TRUE(p4 != p3) << "Expected to get a different object.";
  }
  flag.clear();
  CUDA_CALL(cudaStreamSynchronize(ssync));
}


TEST(PerStreamPool, Massive) {
  std::atomic_flag flag;
  flag.test_and_set();

  int N = 100;
  int niter = 10;
  std::vector<CUDAStream> s(N);
  std::vector<int *> p1(N), p2(N);

  for (int  i = 0; i < N; i++) {
    s[i] = CUDAStream::Create(true);
  }

  CUDAStream ssync = CUDAStream::Create(true);
  CUDAEvent e = CUDAEvent::Create();
  PerStreamPool<int> pool;

  CUDA_CALL(cudaLaunchHostFunc(ssync, wait_func, &flag));
  CUDA_CALL(cudaEventRecord(e, ssync));  // this event is recorded, but not reached, because this
                                         // stream is waiting for a spinning host function

  volatile bool failure = false;
  std::vector<std::thread> t(N);
  for (int i = 0; i < N; i++) {
    t[i] = std::thread([&, i]() {
      for (int j = 0; j < niter; j++) {
        if (auto lease = pool.Get(s[i])) {
          if (j == 0) {
            p1[i] = lease;
            CUDA_CALL(cudaStreamWaitEvent(s[i], e, 0));  // block s[i]
          } else {
            if (lease != p1[i]) {
              std::cerr << "Failure in worker thread " << i
                        << ": object not reused on same stream.";
              failure = true;
              break;
            }
          }
        }
      }
    });
  }

  for (int i = 0; i < N; i++)
    t[i].join();
  EXPECT_FALSE(failure) << "Failure in worker thread";

  flag.clear();
  CUDA_CALL(cudaStreamSynchronize(ssync));

  std::sort(p1.begin(), p1.end());
  for (int i = 1; i < N; i++)
    EXPECT_NE(p1[i], p1[i-1]) << "Duplicate object found - this shouldn't have happened";

  for (int i = 0; i < N; i++)
  CUDA_CALL(cudaStreamSynchronize(s[i]));

  CUDA_CALL(cudaLaunchHostFunc(ssync, wait_func, &flag));
  CUDA_CALL(cudaEventRecord(e, ssync));  // this event is recorded, but not reached, because this
                                         // stream is waiting for a spinning host function

  for (int i = 0; i < N; i++) {
    auto lease = pool.Get(s[i]);
    CUDA_CALL(cudaStreamWaitEvent(s[i], e, 0));  // block s[i]
    p2[i] = lease;
  }

  flag.clear();
  CUDA_CALL(cudaStreamSynchronize(ssync));

  std::sort(p2.begin(), p2.end());
  EXPECT_EQ(p1, p2) << "Should reuse all objects";
}

}  // namespace dali
