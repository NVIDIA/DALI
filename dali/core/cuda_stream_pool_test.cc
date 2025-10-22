// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/mm/memory.h"

namespace dali {

class CUDAStreamPoolTest : public ::testing::Test {
 public:
  void TestPutGet() {
    int devices = 0;
    (void)cudaGetDeviceCount(&devices);
    if (devices == 0) {
      (void)cudaGetLastError();  // No CUDA devices - we don't care about the error
      GTEST_SKIP();
    }

    CUDAStreamPool pool;

    vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
      threads.emplace_back([&]() {
        std::mt19937_64 rng;
        std::uniform_int_distribution<int> dev_dist(0, devices-1);
        std::uniform_int_distribution<int> release_method(5);
        for (int i = 0; i < 10000; i++) {
          int device_id = dev_dist(rng);
          CUDAStreamLease s = pool.Get(device_id);  // not current device (in general)
          CUDAStreamLease stream = std::move(s);  // check move constructor
          ASSERT_EQ(DeviceFromStream(stream), device_id)
            << "CUDAStreamPool mixed streams from different devices";
          ASSERT_EQ(cudaSuccess, cudaSetDevice(device_id));
          ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
          switch (release_method(rng)) {
            case 1:
              stream.reset();
              break;
            case 2:
              pool.Put(stream.release());
              break;
            case 3:
              pool.Put(stream.release(), device_id);
              break;
            case 4:
              stream.release().reset();
              break;
            default:
              // just go out of scope
              break;
          }
        }
      });
    }
    for (auto &t : threads)
      t.join();
    ASSERT_EQ(0, pool.lease_count_.load());
  }
};

namespace test {

TEST_F(CUDAStreamPoolTest, PutGet) {
  TestPutGet();
}

TEST_F(CUDAStreamPoolTest, GetSameNotBusy) {
    CUDAStreamPool pool;
    auto s = pool.Get();
    ASSERT_EQ(cudaStreamQuery(s), cudaSuccess);
    cudaStream_t s1 = s.get();
    s.reset();
    s = pool.Get();
    EXPECT_EQ(s.get(), s1);
}

TEST_F(CUDAStreamPoolTest, GetNotBusy) {
    CUDAStreamPool pool;
    auto s = pool.Get();
    cudaStream_t s1 = s.get();
    ASSERT_EQ(cudaStreamQuery(s), cudaSuccess);
    const int N = 16 << 20;
    auto mem = mm::alloc_raw_async_unique<char, mm::memory_kind::device>(N, s, s, 256);
    for (int i = 0; i < 128; i++)
      CUDA_CALL(cudaMemsetAsync(mem.get(), i, N, s));
    s.reset();
    s = pool.Get();
    EXPECT_EQ(cudaStreamQuery(s), cudaSuccess);
    s.reset();
    CUDA_CALL(cudaStreamSynchronize(s1));  // This stream should still live in the pool
    mem.reset();
    pool.Purge();
}

}  // namespace test
}  // namespace dali
