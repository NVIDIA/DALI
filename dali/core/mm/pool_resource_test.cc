// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <algorithm>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include <chrono>
#include "dali/core/mm/mm_test_utils.h"
#include "dali/core/mm/pool_resource.h"
#include "dali/core/spinlock.h"

namespace dali {
namespace mm {
namespace test {

template <typename FreeList>
void TestPoolResource(int num_iter) {
  test_host_resource upstream;
  {
    auto opt = default_host_pool_opts();
    opt.max_upstream_alignment = 32;  // force the use of overaligned upstream allocations
    pool_resource<memory_kind::host, FreeList, detail::dummy_lock>
      pool(&upstream, opt);
    std::mt19937_64 rng(12345);
    std::bernoulli_distribution is_free(0.4);
    std::uniform_int_distribution<int> align_dist(0, 8);  // alignment anywhere from 1B to 256B
    std::poisson_distribution<int> size_dist(128);
    struct allocation {
      void *ptr;
      size_t size, alignment;
      size_t fill;
    };
    std::vector<allocation> allocs;

    for (int i = 0; i < num_iter; i++) {
      if (i == num_iter / 2)
        pool.release_unused();
      if (is_free(rng) && !allocs.empty()) {
        auto idx = rng() % allocs.size();
        allocation a = allocs[idx];
        CheckFill(a.ptr, a.size, a.fill);
        pool.deallocate(a.ptr, a.size, a.alignment);
        std::swap(allocs[idx], allocs.back());
        allocs.pop_back();
      } else {
        allocation a;
        a.size = std::max(1, std::min(size_dist(rng), 1<<24));
        a.alignment = 1 << align_dist(rng);
        a.fill = rng();
        a.ptr = pool.allocate(a.size, a.alignment);
        ASSERT_TRUE(detail::is_aligned(a.ptr, a.alignment));
        Fill(a.ptr, a.size, a.fill);
        allocs.push_back(a);
      }
    }

    for (auto &a : allocs) {
      CheckFill(a.ptr, a.size, a.fill);
      pool.deallocate(a.ptr, a.size, a.alignment);
    }
    allocs.clear();
  }
  upstream.check_leaks();
}

TEST(MMPoolResource, Coalescing) {
  TestPoolResource<coalescing_free_list>(10000);
}

/* TODO(michalz): Unlock when pool resource can work with best_fit_free_tree
TEST(MMPoolResource, BestFitFreeTree) {
  TestPoolResource<best_fit_free_tree>(100000);
}
*/

TEST(MMPoolResource, CoalescingFreeTree) {
  TestPoolResource<coalescing_free_tree>(100000);
}

TEST(MMPoolResource, ReturnToUpstream) {
  cudaDeviceProp device_prop;
  CUDA_CALL(cudaGetDeviceProperties(&device_prop, 0));
  if (device_prop.integrated) {
      GTEST_SKIP() << "GPU and CPU memory are shared. Overallocating taunts OOM killer";
  }
  test_device_resource upstream;
  {
    pool_resource<memory_kind::device, coalescing_free_tree, detail::dummy_lock>
      pool(&upstream);
    size_t size = 1<<28;  // 256M
    for (;;) {
      try {
        void *mem = pool.allocate(size);
        pool.deallocate(mem, size);
      } catch (const std::bad_alloc &) {
        EXPECT_EQ(upstream.get_current_size(), 0);
        break;
      }
      if (upstream.get_num_deallocs() > 0)
        break;  // deallocation to upstream detected - test passed
      size *= 2;
      if (size == 0) {
        FAIL() << "Reached maximum possible size and there was no out-of-memory error and "
                  "no release to the upstream.";
      }
    }
  }
  upstream.check_leaks();
}

}  // namespace test
}  // namespace mm
}  // namespace dali
