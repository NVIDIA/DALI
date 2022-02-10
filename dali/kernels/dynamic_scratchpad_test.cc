// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/dynamic_scratchpad.h"  // NOLINT
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include "dali/core/cuda_utils.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/mm/memory.h"

namespace dali {
namespace kernels {
namespace test {

/**
 * @brief Tests basic dynamic scratchpad functioning
 *
 * This test checks that:
 * - the memory is usable and accessible on the right backend
 * - the pinned memory block is released in stream order, which prevents
 *   immediate reuse if the stream is still running
 * - it makes multiple attempts to catch the stream still running
 */
TEST(DynamicScratchpad, BasicTest) {
  const int N = 64 << 10;  // 64 KiB

  std::vector<char> in(N);
  for (int i = 0; i < N; i++)
    in[i] = i + 42;  // so it doesn't start or end with 0

  auto stream = CUDAStreamPool::instance().Get();
  auto dev = mm::alloc_raw_unique<char, mm::memory_kind::device>(N);
  int max_attempts = 1000;
  bool was_running = false;
  for (int attempt = 0; attempt < max_attempts; attempt++) {
    char *pinned;
    {
      DynamicScratchpad scratch({}, AccessOrder(stream));
      pinned = scratch.Allocate<mm::memory_kind::pinned, char>(N);
      memcpy(pinned, in.data(), N);
      CUDA_CALL(cudaMemcpyAsync(dev.get(), pinned, N, cudaMemcpyHostToDevice, stream));
    }
    auto out = mm::alloc_raw_unique<char, mm::memory_kind::pinned>(N);
    bool running = false;
    if (was_running) {
      CUDA_CALL(cudaStreamSynchronize(stream));
    } else {
      running = cudaStreamQuery(stream) == cudaErrorNotReady;
      if (running)
        was_running = true;
    }
    ASSERT_TRUE(out.get() + N < pinned || out.get() >= pinned + N || !running);
    CUDA_CALL(cudaMemcpyAsync(out.get(), dev.get(), N, cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));
    ASSERT_EQ(memcmp(in.data(), out.get(), N), 0);
    if (was_running && !running)
      break;
  }
  if (!was_running)
    std::cerr << "Warning: Test incomplete - the stream was never caught still running"
              << std::endl;
}

inline void ProcessResults(vector<double> &times, const string &header) {
  std::sort(times.begin(), times.end());
  double sum = std::accumulate(times.begin(), times.end(), 0);
  auto b98 = times.begin() + times.size()/100;
  auto e98 = times.end() - times.size()/100;
  double sum98 = std::accumulate(b98, e98, 0);
  std::cout << header << "\n"
            << "Median time:            " << times[times.size()/2] << " ns\n"
            << "90th percentile:        " << times[times.size()*90/100] << " ns\n"
            << "99th percentile:        " << times[times.size()*99/100] << " ns\n"
            << "Mean time:              " << sum/times.size() << " ns\n"
            << "Mean time (middle 98%): " << sum98/(e98-b98) << " ns\n";
}

TEST(DynamicScratchpad, Perf) {
  std::poisson_distribution size_dist(1024);  // 1 KiB average
  int max_size = 64 << 20;  // 64 MiB max
  std::uniform_int_distribution<> num_dist(1, 100);

  std::mt19937_64 rng(1234);

  auto stream1 = CUDAStreamPool::instance().Get();
  auto stream2 = CUDAStreamPool::instance().Get();
  cudaStream_t streams[] = { stream1, stream2 };

  int max_attempts = 100000;

  const int nkinds = static_cast<int>(mm::memory_kind_id::count);
  std::vector<double> alloc_times[nkinds];
  std::vector<double> destroy_times;
  for (auto &v : alloc_times)
    v.reserve(max_attempts*100);
  destroy_times.reserve(max_attempts);

  for (int attempt = 0; attempt < max_attempts; attempt++) {
    auto s = streams[attempt % 2];
    std::aligned_storage_t<sizeof(DynamicScratchpad), alignof(DynamicScratchpad)> scratch_placement;
    auto *scratch = new(&scratch_placement) DynamicScratchpad({}, AccessOrder(s));
    for (int k = 0; k < nkinds; k++) {
      auto kind = static_cast<mm::memory_kind_id>(k);
      if (kind == mm::memory_kind_id::managed)
        continue;
      int n = num_dist(rng);
      for (int i = 0; i < n; i++) {
        size_t size = std::min(size_dist(rng), max_size);
        auto s = std::chrono::high_resolution_clock::now();
        scratch->Alloc(kind, size, alignof(std::max_align_t));
        auto e = std::chrono::high_resolution_clock::now();
        alloc_times[k].push_back((e-s).count());
      }
    }
    {
      auto s = std::chrono::high_resolution_clock::now();
      scratch->DynamicScratchpad::~DynamicScratchpad();
      auto e = std::chrono::high_resolution_clock::now();
      destroy_times.push_back((e-s).count());
    }
  }

  const char *names[] = { "host", "pinned", "device", "managed" };

  for (int k = 0; k < nkinds; k++) {
    if (k == mm::memory_kind_id::managed)
      continue;
    ProcessResults(alloc_times[k],
                   make_string("Allocation performance for ", names[k], " memory"));
  }

  ProcessResults(destroy_times, "Scratchpad destruction time");
}


}  // namespace test
}  // namespace kernels
}  // namespace dali
