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

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "dali/core/mm/async_pool.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/mm/mm_test_utils.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/mm/cuda_vm_resource.h"

#if CUDA_VERSION >= 11020
#include "dali/core/mm/malloc_resource.h"
#endif

namespace dali {
namespace mm {

struct GPUHog {
  ~GPUHog() {
    if (mem) {
      CUDA_DTOR_CALL(cudaFree(mem));
      mem = nullptr;
    }
  }

  void init() {
    if (!mem)
      CUDA_CALL(cudaMalloc(&mem, size));
  }

  void run(cudaStream_t stream, int count = 1) {
    for (int i = 0; i < count; i++) {
      CUDA_CALL(cudaMemsetAsync(mem, i+1, size, stream));
    }
  }

  uint8_t *mem = nullptr;
  size_t size = 16<<20;
};

TEST(MMAsyncPool, SingleStreamReuse) {
  GPUHog hog;
  hog.init();

  CUDAStream stream = CUDAStream::Create(true);
  test::test_device_resource upstream;

  async_pool_resource<memory_kind::device> pool(&upstream);
  stream_view sv(stream);
  int size1 = 1<<20;
  void *ptr = pool.allocate_async(size1, sv);
  hog.run(stream, 2);
  pool.deallocate_async(ptr, size1, sv);
  void *p2 = pool.allocate_async(size1, sv);
  CUDA_CALL(cudaStreamSynchronize(stream));
  EXPECT_EQ(ptr, p2);
}

TEST(MMAsyncPool, TwoStream) {
  mm::test::test_device_resource upstream;
  CUDAStream s1 = CUDAStream::Create(true);
  CUDAStream s2 = CUDAStream::Create(true);
  stream_view sv1(s1);
  stream_view sv2(s2);

  GPUHog hog;
  hog.init();
  const int min_success = 10;
  const int max_not_busy = 100;
  int stream_not_busy = 0;
  int success = 0;
  while (success < min_success) {
    async_pool_resource<memory_kind::device> pool(&upstream);
    void *p1 = pool.allocate_async(1000, sv1);
    hog.run(s1);
    pool.deallocate_async(p1, 1000, sv1);
    void *p2 = pool.allocate_async(1000, sv2);
    void *p3 = pool.allocate_async(1000, sv1);
    cudaError_t e = cudaStreamQuery(s1);
    if (e != cudaErrorNotReady) {
      std::cerr << "Stream s1 finished before attempt to allocate on s2 was made - retrying\n";
      CUDA_CALL(cudaGetLastError());
      if (++stream_not_busy > max_not_busy) {
        FAIL() << "Stream s1 finished - test unreliable.";
      }
      continue;
    }
    stream_not_busy = 0;
    ASSERT_NE(p1, p2);
    ASSERT_EQ(p1, p3);
    CUDA_CALL(cudaStreamSynchronize(s1));
    success++;
    CUDA_CALL(cudaStreamSynchronize(s2));
  }
  std::cerr << "Peak consumption:     " << upstream.get_peak_size() << " bytes\n";
  std::cerr << "Upstream allocations: " << upstream.get_num_allocs() << std::endl;
  upstream.check_leaks();
}

namespace {

__global__ void Check(const void *ptr, size_t size, uint8_t fill, int *failures) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (static_cast<const uint8_t*>(ptr)[idx] != fill)
      atomicAdd(failures, 1);
  }
}

struct block {
  void *ptr;
  size_t size, alignment;
  uint8_t fill;
  cudaStream_t stream;
};

template <typename Pool, typename Mutex>
void AsyncPoolTest(Pool &pool, std::vector<block> &blocks, Mutex &mtx, CUDAStream &stream,
                   int max_iters = 20000, bool use_hog = false) {
  stream_view sv(stream);
  std::mt19937_64 rng(12345);
  std::poisson_distribution<> size_dist(1024);
  std::uniform_int_distribution<> align_log_dist(1, 12);
  const int max_size = 1 << 20;
  std::uniform_int_distribution<> sync_dist(10, 10);
  std::bernoulli_distribution action_dist;
  std::bernoulli_distribution hog_dist(0.05f);
  std::uniform_int_distribution<> fill_dist(1, 255);
  DeviceBuffer<int> failure_buf;
  int failures = 0;
  failure_buf.from_host(&failures, 1, sv.get());
  GPUHog hog;
  if (use_hog)
    hog.init();
  int hogs = 0;
  int max_hogs = sync_dist(rng);
  CUDAEvent event = CUDAEvent::Create();
  for (int i = 0; i < max_iters; i++) {
    if constexpr (std::is_base_of_v<mm::pool_resource_base<typename Pool::memory_kind>, Pool>) {
      if (i == max_iters / 2)
        pool.release_unused();
    }

    if (use_hog && hog_dist(rng)) {
      if (hogs++ > max_hogs) {
        CUDA_CALL(cudaStreamSynchronize(stream));
        max_hogs = sync_dist(rng);
        hogs = 0;
      }
      hog.run(stream);
    }
    if (action_dist(rng) || blocks.empty()) {
      size_t size, alignment;
      do {
        size = size_dist(rng);
      } while (size > max_size);
      uint8_t fill = fill_dist(rng);
      alignment = 1 << align_log_dist(rng);
      void *ptr = stream ? pool.allocate_async(size, alignment, sv)
                         : pool.allocate(size, alignment);
      ASSERT_TRUE(mm::detail::is_aligned(ptr, alignment));
      CUDA_CALL(cudaMemsetAsync(ptr, fill, size, stream));
      {
        std::lock_guard<Mutex> guard(mtx);
        (void)guard;  // for dummy mutexes
        blocks.push_back({ ptr, size, alignment, fill, stream });
      }
    } else {
      block blk;
      {
        std::lock_guard<Mutex> guard(mtx);
        (void)guard;  // for dummy mutexes
        if (blocks.empty())
          continue;
        int i = std::uniform_int_distribution<>(0, blocks.size()-1)(rng);
        std::swap(blocks[i], blocks.back());
        blk = blocks.back();
        blocks.pop_back();
      }
      if (blk.stream != stream) {
        if (stream) {
          CUDA_CALL(cudaEventRecord(event, blk.stream));
          CUDA_CALL(cudaStreamWaitEvent(stream, event, 0));
        } else {
          CUDA_CALL(cudaStreamSynchronize(blk.stream));
        }
      }
      Check<<<div_ceil(blk.size, 1024), 1024, 0, stream>>>(
            blk.ptr, blk.size, blk.fill, failure_buf);
      if (stream) {
        pool.deallocate_async(blk.ptr, blk.size, blk.alignment, sv);
      } else {
        CUDA_CALL(cudaStreamSynchronize(stream));
        pool.deallocate(blk.ptr, blk.size, blk.alignment);
      }
    }
  }
  copyD2H<int>(&failures, failure_buf, 1, AccessOrder(stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
  ASSERT_EQ(failures, 0);
}

}  // namespace

TEST(MMAsyncPool, SingleStreamRandom) {
  CUDAStream stream = CUDAStream::Create(true);
  test::test_device_resource upstream;

  {
    async_pool_resource<memory_kind::device> pool(&upstream);
    std::vector<block> blocks;
    detail::dummy_lock mtx;
    AsyncPoolTest(pool, blocks, mtx, stream);
  }

  CUDA_CALL(cudaStreamSynchronize(stream));
  std::cerr << "Peak consumption:     " << upstream.get_peak_size() << " bytes\n";
  std::cerr << "Upstream allocations: " << upstream.get_num_allocs() << std::endl;
  upstream.check_leaks();
}

TEST(MMAsyncPool, MultiThreadedSingleStreamRandom) {
  CUDAStream stream = CUDAStream::Create(true);
  mm::test::test_device_resource upstream;
  {
    std::vector<block> blocks;
    std::mutex mtx;

    async_pool_resource<memory_kind::device> pool(&upstream);

    std::vector<std::thread> threads;

    for (int t = 0; t < 10; t++) {
      threads.push_back(std::thread([&]() {
        AsyncPoolTest(pool, blocks, mtx, stream);
      }));
    }
    for (auto &t : threads)
      t.join();
  }
  CUDA_CALL(cudaStreamSynchronize(stream));
  std::cerr << "Peak consumption:     " << upstream.get_peak_size() << " bytes\n";
  std::cerr << "Upstream allocations: " << upstream.get_num_allocs() << std::endl;
  upstream.check_leaks();
}

TEST(MMAsyncPool, MultiThreadedMultiStreamRandom) {
  mm::test::test_device_resource upstream;
  {
    async_pool_resource<memory_kind::device> pool(&upstream);

    std::vector<std::thread> threads;

    for (int t = 0; t < 10; t++) {
      threads.push_back(std::thread([&]() {
        CUDAStream stream = CUDAStream::Create(true);
        std::vector<block> blocks;
        detail::dummy_lock mtx;
        AsyncPoolTest(pool, blocks, mtx, stream);
        CUDA_CALL(cudaStreamSynchronize(stream));
      }));
    }
    for (auto &t : threads)
      t.join();
  }
  std::cerr << "Peak consumption:     " << upstream.get_peak_size() << " bytes\n";
  std::cerr << "Upstream allocations: " << upstream.get_num_allocs() << std::endl;
  upstream.check_leaks();
}

TEST(MMAsyncPool, MultiStreamRandomWithGPUHogs) {
  mm::test::test_device_resource upstream;
  {
    async_pool_resource<memory_kind::device> pool(&upstream, false);

    std::vector<std::thread> threads;

    for (int t = 0; t < 10; t++) {
      threads.push_back(std::thread([&]() {
        // 0-th thread uses null stream, which triggers non-async API usage
        CUDAStream stream = t ? CUDAStream::Create(true) : CUDAStream();
        std::vector<block> blocks;
        detail::dummy_lock mtx;
        AsyncPoolTest(pool, blocks, mtx, stream, 20000, true);
        CUDA_CALL(cudaStreamSynchronize(stream));
      }));
    }
    for (auto &t : threads)
      t.join();
  }
  std::cerr << "Peak consumption:     " << upstream.get_peak_size() << " bytes\n";
  std::cerr << "Upstream allocations: " << upstream.get_num_allocs() << std::endl;
  upstream.check_leaks();
}


TEST(MMAsyncPool, CrossStream) {
  mm::test::test_device_resource upstream;
  {
    async_pool_resource<memory_kind::device> pool(&upstream, false);

    std::vector<std::thread> threads;
    std::vector<CUDAStream> streams;

    std::vector<block> blocks;
    std::mutex mtx;

    const int N = 10;
    streams.resize(N);
    for (int t = 0; t < N; t++) {
      if (t != 0)  // keep empty stream at index 0 to mix sync/async allocations
        streams[t] = CUDAStream::Create(true);
      threads.push_back(std::thread([&, t]() {
        AsyncPoolTest(pool, blocks, mtx, streams[t]);
        CUDA_CALL(cudaStreamSynchronize(streams[t]));
      }));
    }
    for (auto &t : threads)
      t.join();
  }
  std::cerr << "Peak consumption:     " << upstream.get_peak_size() << " bytes\n";
  std::cerr << "Upstream allocations: " << upstream.get_num_allocs() << std::endl;
  upstream.check_leaks();
}

TEST(MMAsyncPool, CrossStreamWithHogs) {
  mm::test::test_device_resource upstream;
  {
    async_pool_resource<memory_kind::device> pool(&upstream);

    std::vector<std::thread> threads;
    std::vector<CUDAStream> streams;

    std::vector<block> blocks;
    std::mutex mtx;

    const int N = 10;
    streams.resize(N);
    for (int t = 0; t < N; t++) {
      if (t != 0)  // keep empty stream at index 0 to mix sync/async allocations
        streams[t] = CUDAStream::Create(true);
      threads.push_back(std::thread([&, t]() {
        AsyncPoolTest(pool, blocks, mtx, streams[t], 10000, true);
        CUDA_CALL(cudaStreamSynchronize(streams[t]));
      }));
    }
    for (auto &t : threads)
      t.join();
  }
  std::cerr << "Peak consumption:     " << upstream.get_peak_size() << " bytes\n";
  std::cerr << "Upstream allocations: " << upstream.get_num_allocs() << std::endl;
  upstream.check_leaks();
}

class MMAsyncPoolTest : public ::testing::Test {
 public:
  template <typename MememoryResource>
  void MultiThreadedSingleStreamRandom() {
    CUDAStream stream = CUDAStream::Create(true);
    {
      std::vector<block> blocks;
      std::mutex mtx;

      MememoryResource pool;

      std::vector<std::thread> threads;

      for (int t = 0; t < 10; t++) {
        threads.push_back(std::thread([&]() {
          AsyncPoolTest(pool, blocks, mtx, stream);
        }));
      }
      for (auto &t : threads)
        t.join();
    }
  }

  template <typename MemoryResource>
  void MultiThreadedMultiStreamRandom() {
    MemoryResource pool;

    std::vector<std::thread> threads;

    for (int t = 0; t < 10; t++) {
      threads.push_back(std::thread([&]() {
        CUDAStream stream = CUDAStream::Create(true);
        std::vector<block> blocks;
        detail::dummy_lock mtx;
        AsyncPoolTest(pool, blocks, mtx, stream);
        CUDA_CALL(cudaStreamSynchronize(stream));
      }));
    }
    for (auto &t : threads)
      t.join();
  }

  template <typename MemoryResource>
  void MultiStreamRandomWithGPUHogs() {
    MemoryResource pool;

    std::vector<std::thread> threads;

    for (int t = 0; t < 10; t++) {
      threads.push_back(std::thread([&]() {
        // 0-th thread uses null stream, which triggers non-async API usage
        CUDAStream stream = t ? CUDAStream::Create(true) : CUDAStream();
        std::vector<block> blocks;
        detail::dummy_lock mtx;
        AsyncPoolTest(pool, blocks, mtx, stream, 20000, true);
        CUDA_CALL(cudaStreamSynchronize(stream));
      }));
    }
    for (auto &t : threads)
      t.join();
  }

  template <typename MemoryResource>
  void CrossStream() {
    async_pool_resource<memory_kind::device, cuda_vm_resource> pool;

    std::vector<std::thread> threads;
    std::vector<CUDAStream> streams;

    std::vector<block> blocks;
    std::mutex mtx;

    const int N = 10;
    streams.resize(N);
    for (int t = 0; t < N; t++) {
      if (t != 0)  // keep empty stream at index 0 to mix sync/async allocations
        streams[t] = CUDAStream::Create(true);
      threads.push_back(std::thread([&, t]() {
        AsyncPoolTest(pool, blocks, mtx, streams[t]);
        CUDA_CALL(cudaStreamSynchronize(streams[t]));
      }));
    }
    for (auto &t : threads)
      t.join();
  }

  template <typename MemoryResource>
  void CrossStreamWithHogs() {
    MemoryResource pool;

    std::vector<std::thread> threads;
    std::vector<CUDAStream> streams;

    std::vector<block> blocks;
    std::mutex mtx;

    const int N = 10;
    streams.resize(N);
    for (int t = 0; t < N; t++) {
      if (t != 0)  // keep empty stream at index 0 to mix sync/async allocations
        streams[t] = CUDAStream::Create(true);
      threads.push_back(std::thread([&, t]() {
        AsyncPoolTest(pool, blocks, mtx, streams[t], 10000, true);
        CUDA_CALL(cudaStreamSynchronize(streams[t]));
      }));
    }
    for (auto &t : threads)
      t.join();
  }
};

#if DALI_USE_CUDA_VM_MAP

TEST_F(MMAsyncPoolTest, VM_MultiThreadedSingleStreamRandom) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "Virtual memory management API is not supported on this machine.";

  using MR = async_pool_resource<memory_kind::device, cuda_vm_resource>;
  this->MultiThreadedSingleStreamRandom<MR>();
}

TEST_F(MMAsyncPoolTest, VM_MultiThreadedMultiStreamRandom) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "Virtual memory management API is not supported on this machine.";

  using MR = async_pool_resource<memory_kind::device, cuda_vm_resource>;
  this->MultiThreadedMultiStreamRandom<MR>();
}

TEST_F(MMAsyncPoolTest, VM_MultiStreamRandomWithGPUHogs) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "Virtual memory management API is not supported on this machine.";

  using MR = async_pool_resource<memory_kind::device, cuda_vm_resource>;
  this->MultiStreamRandomWithGPUHogs<MR>();
}

TEST_F(MMAsyncPoolTest, VM_CrossStream) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "Virtual memory management API is not supported on this machine.";

  using MR = async_pool_resource<memory_kind::device, cuda_vm_resource>;
  this->CrossStream<MR>();
}

TEST_F(MMAsyncPoolTest, VM_CrossStreamWithHogs) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "Virtual memory management API is not supported on this machine.";

  using MR = async_pool_resource<memory_kind::device, cuda_vm_resource>;
  this->CrossStreamWithHogs<MR>();
}

#endif

#if CUDA_VERSION >= 11020

TEST_F(MMAsyncPoolTest, cudaMallocAsync_MultiThreadedSingleStreamRandom) {
  if (!cuda_malloc_async_memory_resource::is_supported())
    GTEST_SKIP() << "cudaMallocAsync not supported";

  using MR = cuda_malloc_async_memory_resource;
  this->MultiThreadedSingleStreamRandom<MR>();
}

TEST_F(MMAsyncPoolTest, cudaMallocAsync_MultiThreadedMultiStreamRandom) {
  if (!cuda_malloc_async_memory_resource::is_supported())
    GTEST_SKIP() << "cudaMallocAsync not supported";

  using MR = cuda_malloc_async_memory_resource;
  this->MultiThreadedMultiStreamRandom<MR>();
}

TEST_F(MMAsyncPoolTest, cudaMallocAsync_MultiStreamRandomWithGPUHogs) {
  if (!cuda_malloc_async_memory_resource::is_supported())
    GTEST_SKIP() << "cudaMallocAsync not supported";

  using MR = cuda_malloc_async_memory_resource;
  this->MultiStreamRandomWithGPUHogs<MR>();
}

TEST_F(MMAsyncPoolTest, cudaMallocAsync_CrossStream) {
  if (!cuda_malloc_async_memory_resource::is_supported())
    GTEST_SKIP() << "cudaMallocAsync not supported";

  using MR = cuda_malloc_async_memory_resource;
  this->CrossStream<MR>();
}

TEST_F(MMAsyncPoolTest, cudaMallocAsync_CrossStreamWithHogs) {
  if (!cuda_malloc_async_memory_resource::is_supported())
    GTEST_SKIP() << "cudaMallocAsync not supported";

  using MR = cuda_malloc_async_memory_resource;
  this->CrossStreamWithHogs<MR>();
}

#endif

}  // namespace mm
}  // namespace dali
