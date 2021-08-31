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

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "dali/core/mm/async_pool.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/mm/mm_test_utils.h"
#include "dali/core/cuda_stream.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace mm {
namespace test {

TEST(MMPinnedAlloc, StageCopy) {
  test_pinned_resource upstream;
  {
    CUDAStream stream = CUDAStream::Create(true);
    stream_view sv(stream);
    async_pool_resource<memory_kind::pinned> pool(&upstream);
    std::mt19937_64 rng;
    const int N = 1<<20;
    vector<uint8_t> pattern(N), copy_back(N);
    DeviceBuffer<uint8_t> dev_buf;
    dev_buf.resize(N);
    UniformRandomFill(pattern, rng, 0, 255);
    void *mem1 = pool.allocate(N);
    memcpy(mem1, pattern.data(), N);
    CUDA_CALL(cudaMemcpyAsync(dev_buf, mem1, N, cudaMemcpyHostToDevice, stream));
    pool.deallocate_async(mem1, N, sv);
    void *mem2 = pool.allocate_async(N, sv);
    EXPECT_EQ(mem1, mem2);
    CUDA_CALL(cudaMemcpyAsync(copy_back.data(), dev_buf, N, cudaMemcpyDeviceToHost, stream));
    pool.deallocate_async(mem1, N, sv);
    CUDA_CALL(cudaStreamSynchronize(stream));
    EXPECT_EQ(pattern, copy_back);
  }
  upstream.check_leaks();
}

TEST(MMPinnedAlloc, SyncAndSteal) {
  test_pinned_resource upstream;
  {
    CUDAStream s1, s2;
    s1 = CUDAStream::Create(true);
    s2 = CUDAStream::Create(true);
    stream_view sv1(s1), sv2(s2);
    const int N = 1<<24;
    async_pool_resource<memory_kind::pinned> pool(&upstream, true);
    void *mem1 = pool.allocate_async(N, sv1);
    CUDA_CALL(cudaMemsetAsync(mem1, 0, N, s1));
    pool.deallocate_async(mem1, N, sv1);
    // We've requested a large chunk (16MiB) of memory - that memory is not going
    // to be readily available, but the pool is configured with "avoid upstream" option
    // and therefore will wait for the pending deallocations to complete - this is still
    // lighter than calling cudaMallocHost, which would implicitly synchronize all devices,
    // not just some streams.
    void *mem2 = pool.allocate_async(N, sv2);
    auto e = cudaStreamQuery(s1);
    EXPECT_NE(e, cudaErrorNotReady) << "Synchronization should have occurred";
    if (e != cudaErrorNotReady) {
      CUDA_CALL(cudaGetLastError());
    }
    EXPECT_EQ(mem1, mem2) << "Memory should have been stolen from the stream1 after it's finished";
    pool.deallocate_async(mem2, N, sv2);
  }
  upstream.check_leaks();
}

TEST(MMPinnedAlloc, SyncCrossDevice) {
  test_pinned_resource upstream;
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  } else {
    CUDAStream s1, s2;
    DeviceGuard dg(0);
    s1 = CUDAStream::Create(true);
    CUDA_CALL(cudaSetDevice(1));
    s2 = CUDAStream::Create(true);
    CUDA_CALL(cudaSetDevice(0));
    stream_view sv1(s1), sv2(s2);
    const int N = 1<<24;
    async_pool_resource<memory_kind::pinned> pool(&upstream, true);
    void *mem1 = pool.allocate_async(N, sv1);
    CUDA_CALL(cudaMemsetAsync(mem1, 0, N, s1));
    pool.deallocate_async(mem1, N, sv1);
    // We've requested a large chunk (16MiB) of memory - that memory is not going
    // to be readily available, but the pool is configured with "avoid upstream" option
    // and therefore will wait for the pending deallocations to complete - this is still
    // lighter than calling cudaMallocHost, which would implicitly synchronize all devices,
    // not just some streams.
    void *mem2 = pool.allocate_async(N, sv2);
    auto e = cudaStreamQuery(s1);
    EXPECT_NE(e, cudaErrorNotReady) << "Synchronization should have occurred";
    if (e != cudaErrorNotReady) {
      CUDA_CALL(cudaGetLastError());
    }
    EXPECT_EQ(mem1, mem2) << "Memory should have been stolen from the stream1 after it's finished";
    pool.deallocate_async(mem2, N, sv2);
  }
  upstream.check_leaks();
}

TEST(MMPinnedAlloc, FreeOnAnotherDevice) {
  test_pinned_resource upstream;
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  } else {
    CUDAStream s1, s2;
    DeviceGuard dg(0);
    s1 = CUDAStream::Create(true);
    CUDA_CALL(cudaSetDevice(1));
    s2 = CUDAStream::Create(true);
    CUDA_CALL(cudaSetDevice(0));
    stream_view sv1(s1), sv2(s2);
    const int N = 1<<24;
    async_pool_resource<memory_kind::pinned> pool(&upstream, true);
    void *mem1 = pool.allocate_async(N, sv1);
    CUDA_CALL(cudaMemsetAsync(mem1, 0, N, s1));
    CUDA_CALL(cudaStreamSynchronize(s1));
    // don't set device - it should be inferred from the stream
    pool.deallocate_async(mem1, N, sv2);
    // now set the device and allocate
    CUDA_CALL(cudaSetDevice(1));
    void *mem2 = pool.allocate_async(N, sv2);
    EXPECT_EQ(mem1, mem2) << "Memory should have been moved to stream2 on another device.";
    pool.deallocate_async(mem2, N, sv2);
  }
  upstream.check_leaks();
}


}  // namespace test
}  // namespace mm
}  // namespace dali
