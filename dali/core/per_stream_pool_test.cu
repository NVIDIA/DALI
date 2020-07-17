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
#include <chrono>
#include <thread>
#include "dali/core/cuda_stream.h"

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

  cudaLaunchHostFunc(ssync, wait_func, &flag);
  cudaEventRecord(e, ssync);  // this event is recorded, but not reached, because this stream is
                              // waiting for a spinning host function

  cudaStreamSynchronize(s1);  // make sure that stream has completed its job
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_EQ(p2, p1) << "Expected to get the same object.";
    cudaStreamWaitEvent(s1, e, 0);  // block s1
  }
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_EQ(p2, p1) << "Expected to get the same object, even if job is still pending.";
  }
  flag.clear();  // unblock stream ssync
  cudaStreamSynchronize(ssync);
  cudaStreamSynchronize(s1);
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
  cudaLaunchHostFunc(ssync, wait_func, &flag);
  cudaEventRecord(e, ssync);  // this event is recorded, but not reached, because this stream is
                              // waiting for a spinning host function

  cudaStreamSynchronize(s1);  // make sure that stream has completed its job
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_EQ(p2, p1) << "Expected to get the same object.";
    cudaStreamWaitEvent(s1, e, 0);  // block s1
  }
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_NE(p2, p1) << "Expected to get a new object - job is still pending and reuse disabled.";
  }
  flag.clear();  // unblock stream ssync
  cudaStreamSynchronize(ssync);
  cudaStreamSynchronize(s1);
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

  cudaLaunchHostFunc(ssync, wait_func, &flag);
  cudaEventRecord(e, ssync);  // this event is recorded, but not reached, because this stream is
                              // waiting for a spinning host function

  cudaStreamSynchronize(s1);  // make sure that stream has completed its job
  if (auto lease = pool.Get(s1)) {
    p2 = lease;
    EXPECT_EQ(p2, p1) << "Expected to get the same object.";
    cudaStreamWaitEvent(s1, e, 0);  // block s1
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
  cudaStreamSynchronize(ssync);
  cudaStreamSynchronize(s1);
  cudaStreamSynchronize(s2);
  if (auto lease = pool.Get(s1)) {
    p3 = lease;
    EXPECT_TRUE(p3 == p1 || p3 == p2) << "Expected to get one of the previously objects.";
  }

  cudaLaunchHostFunc(ssync, wait_func, &flag);
  cudaEventRecord(e, ssync);  // this event is recorded, but not reached, because this stream is
                              // waiting for a spinning host function

  if (auto lease = pool.Get(s1)) {
    p4 = lease;
    EXPECT_TRUE(p4 == p3) << "Expected to get one of the previously objects.";
    cudaStreamWaitEvent(s1, e, 0);  // block s1
  }
  if (auto lease = pool.Get(s2)) {
    p4 = lease;
    EXPECT_TRUE(p4 != p3) << "Expected to get a different object.";
  }
  flag.clear();
  cudaStreamSynchronize(ssync);
}

}  // namespace dali
