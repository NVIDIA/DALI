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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <thread>
#include <vector>
#include "dali/core/cuda_error.h"
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_stream.h"

namespace dali {
namespace test {

TEST(EventPoolTest, PutGet) {
  int devices = 0;
  (void)cudaGetDeviceCount(&devices);
  if (devices == 0) {
    (void)cudaGetLastError();  // No CUDA devices - we don't care about the error
    GTEST_SKIP();
  }

  CUDAEventPool pool;

  vector<CUDAStream> streams;
  for (int i = 0; i < devices; i++) {
    streams.push_back(CUDAStream::Create(true, i));
  }

  vector<std::thread> threads;
  for (int i = 0; i < 10; i++) {
    threads.emplace_back([&]() {
      std::mt19937_64 rng;
      std::uniform_int_distribution<int> dev_dist(0, devices-1);
      for (int i = 0; i < 10000; i++) {
        int device_id = dev_dist(rng);
        CUDAEvent event = pool.Get(device_id);  // not current device (in general)
        ASSERT_EQ(cudaSuccess, cudaSetDevice(device_id));
        ASSERT_EQ(cudaSuccess, cudaEventRecord(event, streams[device_id]));
        ASSERT_EQ(cudaSuccess, cudaEventSynchronize(event));
        pool.Put(std::move(event));
      }
    });
  }
  for (auto &t : threads)
    t.join();
}

}  // namespace test
}  // namespace dali
