// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "dali/core/mm/default_resources.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/spinlock.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_error.h"
#include "dali/core/device_guard.h"
#include "dali/test/timing.h"

namespace dali {
namespace mm {
namespace test {

using dali::test::format_time;
using dali::test::perf_timer;
using dali::test::seconds;

void RunBenchmark(mm::async_memory_resource<mm::memory_kind::device> *res,
                  int num_threads,
                  int num_streams) {
  std::vector<CUDAStreamLease> streams;
  streams.reserve(num_streams);
  for (int i = 0; i < num_streams; i++)
    streams.push_back(CUDAStreamPool::instance().Get());
  struct Alloc {
    void *ptr;
    size_t size, alignment;
    AccessOrder order = AccessOrder::host();
  };
  std::vector<Alloc> allocs;


  allocs.reserve(100000);
  spinlock lock;

  perf_timer::duration total_alloc_time = {};
  perf_timer::duration total_dealloc_time = {};
  perf_timer::duration total_async_alloc_time = {};
  perf_timer::duration total_async_dealloc_time = {};
  int64_t total_num_allocs = 0, total_num_deallocs = 0;
  int64_t total_num_async_allocs = 0, total_num_async_deallocs = 0;

  std::vector<std::thread> threads;
  for (int tid = 0; tid < num_threads; tid++) {
    threads.emplace_back([&, tid /* to avoid future bugs */]() {
      (void)tid;  // Silence a terribly ill-advised warning from clang.
      std::mt19937_64 rng;
      std::uniform_int_distribution<int> stream_dist(-1, num_streams - 1);
      std::uniform_real_distribution<float> size_log_dist(4, 24);
      std::bernoulli_distribution action_dist(0.5);

      perf_timer::duration alloc_time = {};
      perf_timer::duration dealloc_time = {};
      perf_timer::duration async_alloc_time = {};
      perf_timer::duration async_dealloc_time = {};
      int64_t num_allocs = 0, num_deallocs = 0;
      int64_t num_async_allocs = 0, num_async_deallocs = 0;

      for (int iter = 0; iter < 100000; iter++) {
        bool is_free = action_dist(rng);
        if (is_free) {
          Alloc alloc;

          {
            // Get an allocation and quickly remove it from `allocs` by swapping it with
            // the last element.

            std::lock_guard g(lock);
            if (allocs.empty())
              continue;
            int idx = std::uniform_int_distribution<int>(0, allocs.size() - 1)(rng);
            alloc = allocs[idx];
            std::swap(allocs[idx], allocs.back());
            allocs.pop_back();
          }
          int stream_idx = stream_dist(rng);
          if (stream_idx < 0) {
            auto start = perf_timer::now();
            res->deallocate(alloc.ptr, alloc.size, alloc.alignment);
            auto end = perf_timer::now();
            dealloc_time += (end-start);
            num_deallocs++;
          } else {
            assert(stream_idx >= 0 && stream_idx < streams.size());
            cudaStream_t stream = streams[stream_idx].get();
            auto start = perf_timer::now();
            res->deallocate_async(alloc.ptr, alloc.size, alloc.alignment, stream);
            auto end = perf_timer::now();
            async_dealloc_time += (end-start);
            num_async_deallocs++;
          }
        } else {  // allocate
          int stream_idx = stream_dist(rng);

          Alloc alloc = {};
          alloc.size = static_cast<int>(powf(2, size_log_dist(rng)));
          alloc.alignment = 256;

          if (stream_idx < 0) {
            alloc.order = AccessOrder::host();
            auto start = perf_timer::now();
            alloc.ptr = res->allocate(alloc.size, alloc.alignment);
            auto end = perf_timer::now();
            alloc_time += (end-start);
            num_allocs++;
          } else {
            assert(stream_idx >= 0 && stream_idx < streams.size());
            cudaStream_t stream = streams[stream_idx].get();
            alloc.order = stream;
            auto start = perf_timer::now();
            alloc.ptr = res->allocate_async(alloc.size, alloc.alignment, stream);
            auto end = perf_timer::now();
            async_alloc_time += (end-start);
            num_async_allocs++;
          }
          {
            std::lock_guard g(lock);
            allocs.push_back(alloc);
          }
        }
      }

      {
        std::lock_guard g(lock);
        total_alloc_time += alloc_time;
        total_dealloc_time += dealloc_time;
        total_num_allocs += num_allocs;
        total_num_deallocs += num_deallocs;

        total_async_alloc_time += async_alloc_time;
        total_async_dealloc_time += async_dealloc_time;
        total_num_async_allocs += num_async_allocs;
        total_num_async_deallocs += num_async_deallocs;
      }
    });
  }

  for (auto &t : threads)
    t.join();

  for (auto &alloc : allocs) {
    res->deallocate(alloc.ptr, alloc.size, alloc.alignment);
  }

  print(std::cout,
    "# allocations:           ", total_num_allocs, "\n"
    "# deallocations:         ", total_num_deallocs, "\n"
    "# async allocations:     ", total_num_async_allocs, "\n"
    "# async deallocations:   ", total_num_async_deallocs, "\n"
    "Allocation time:         ", format_time(seconds(total_alloc_time) / total_num_allocs), "\n"
    "Dellocation time:        ", format_time(seconds(total_dealloc_time) / total_num_allocs), "\n"
    "Async allocation time:   ", format_time(seconds(total_async_alloc_time) / total_num_allocs),
    "\n"
    "Async deallocation time: ", format_time(seconds(total_async_dealloc_time) / total_num_allocs),
    "\n");
}

TEST(MMPerfTest, DefaultGPUAlloc) {
  auto *res = mm::GetDefaultDeviceResource(0);

  RunBenchmark(res, 1, 1);
}

#if CUDA_VERSION >= 11020
TEST(MMPerfTest, CudaMallocAsync) {
  if (!cuda_malloc_async_memory_resource::is_supported())
    GTEST_SKIP() << "cudaMallocAsync not supported";
  cuda_malloc_async_memory_resource res;
  RunBenchmark(&res, 1, 1);
}
#endif

}  // namespace test
}  // namespace mm
}  // namespace dali

