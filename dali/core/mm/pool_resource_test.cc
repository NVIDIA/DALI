// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    pool_resource_base<memory_kind::host, any_context, FreeList, detail::dummy_lock>
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
  test_device_resource upstream;
  {
    pool_resource_base<memory_kind::device, any_context, coalescing_free_tree, detail::dummy_lock>
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

TEST(MMPoolResource, TestBulkDeallocate) {
  test_host_resource upstream;
  {
    pool_options opts;
    opts.min_block_size = 1000000;
    pool_resource_base<memory_kind::host, any_context, coalescing_free_tree, detail::dummy_lock>
      pool(&upstream, opts);
    vector<dealloc_params> params;
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int> align_dist(0, 8);  // alignment anywhere from 1B to 256B
    std::poisson_distribution<int> size_dist(128);
    for (int i = 0; i < 100; i++) {
      dealloc_params par;
      par.bytes = size_dist(rng);
      if (par.bytes > 8192) par.bytes = 8192;  // make sure we fit in the preallocated size
      par.alignment = 1 << align_dist(rng);
      par.ptr = pool.allocate(par.bytes, par.alignment);
      params.push_back(par);
    }
    pool.bulk_deallocate(make_span(params));  // free everything
    for (auto &par : params) {
      void *p = pool.allocate(par.bytes, par.alignment);
      EXPECT_EQ(p, par.ptr);  // now we should get the same pointers
    }
    pool.bulk_deallocate(make_span(params));
  }
  upstream.check_leaks();
}

namespace {

template <typename Kind, typename Context = any_context>
class test_defer_dealloc
    : public deferred_dealloc_pool<Kind, Context, coalescing_free_tree, spinlock> {
 public:
  using pool = deferred_dealloc_pool<Kind, Context, coalescing_free_tree, spinlock>;
  bool ready() const noexcept {
    return this->no_pending_deallocs();
  }

  explicit test_defer_dealloc(memory_resource<Kind, Context> *upstream) : pool(upstream) {}

  void check() {
    // wait up to 1 second for the deferred deallocations to drain
    for (int t = 0; !ready() && t < 100; t++)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_TRUE(ready()) << "Timeout - the resource should be ready by this time.";
    // Now go over all blocks and see if they are all free - that will tell us
    // if all blocks were properly freed.
    for (auto &b : this->blocks_) {
      EXPECT_TRUE(this->free_list_.remove_if_in_list(b.ptr, b.bytes))
        << "All allocations should have been deallocated";
      this->free_list_.put(b.ptr, b.bytes);
    }
  }
};

}  // namespace

TEST(MMPoolResource, DeferredCheckAsync) {
  test_pinned_resource upstream;
  {
    test_defer_dealloc<memory_kind::pinned> pool(&upstream);
    const int max_attempts = 10000;
    bool success = false;
    for (int i = 0; i < max_attempts; i++) {
      int size = 100 * (i + 1);
      void *p1 = pool.allocate(size);
      void *p2 = pool.allocate(size);
      pool.deferred_deallocate(p1, size);
      pool.deferred_deallocate(p2, size);
      bool ready = pool.ready();
      void *p3 = pool.allocate(size);
      if (p1 != p3 && !ready) {
        success = true;  // ok, the deallocation was truly asynchrounous
        pool.deferred_deallocate(p3, size);
        break;
      } else {
        // recheck to avoid race condition
        ready = pool.ready();
        pool.deferred_deallocate(p3, size);
        ASSERT_TRUE(ready) << "The resource returned two identical pointers "
                              "but the deallocation is reported as incomplete.\n";
      }
      pool.check();
    }
    EXPECT_TRUE(success) << "All deallocations finished immediately in "
                         << max_attempts << " attemtps.";
  }
}

namespace {

struct block {
  void *ptr;
  size_t size;
  uint8_t fill;
};

template <typename Pool, typename Mutex>
void PoolTest(Pool &pool, vector<block> &blocks, Mutex &mtx, int max_iters = 20000) {
  std::mt19937_64 rng(12345);
  std::poisson_distribution<> size_dist(1<<20);
  const int max_size = 1 << 26;
  std::bernoulli_distribution action_dist;
  std::bernoulli_distribution flush_dist(0.01);
  std::uniform_int_distribution<> fill_dist(1, 255);
  for (int i = 0; i < max_iters; i++) {
    if (flush_dist(rng))
      pool.flush_deferred();
    if (action_dist(rng) || blocks.empty()) {
      size_t size;
      do {
        size = size_dist(rng);
      } while (size > max_size);
      uint8_t fill = fill_dist(rng);
      void *ptr = pool.allocate(size);
      ASSERT_NE(ptr, nullptr);
      if (size <= 2048) {  // small block - fill it entirely in one go
        memset(ptr, fill, size);
      } else {
        // large block - only fill the beginning and the end in order to save time - this test
        // must be fast or it will fail to detect race conditions.
        memset(ptr, fill, 1024);
        memset(static_cast<char*>(ptr) + size - 1024, fill, 1024);
      }
      {
        std::lock_guard<Mutex> guard(mtx);
        blocks.push_back({ ptr, size, fill });
      }
    } else {
      block blk;
      {
        std::lock_guard<Mutex> guard(mtx);
        if (blocks.empty())
          continue;
        int i = std::uniform_int_distribution<>(0, blocks.size()-1)(rng);
        std::swap(blocks[i], blocks.back());
        blk = blocks.back();
        blocks.pop_back();
      }

      // For small blocks (up to 2 KiB), check the whole block - for larger blocks,
      // just check the first and last 1 KiB.
      size_t part1 = blk.size <= 2048 ? blk.size : 1024;
      size_t part2 = blk.size <= 2048 ? 0 : 1024;

      for (size_t i = 0; i < part1; i++) {
        ASSERT_EQ(static_cast<uint8_t*>(blk.ptr)[i], blk.fill)
          << "Corruption in block " << blk.ptr
          << " at offset " << i;
      }
      for (size_t i = blk.size - part2; i < blk.size; i++) {
        ASSERT_EQ(static_cast<uint8_t*>(blk.ptr)[i], blk.fill)
          << "Corruption in block " << blk.ptr
          << " at offset " << i;
      }

      pool.deallocate(blk.ptr, blk.size);
    }
  }
}

}  // namespace

TEST(MMPoolResource, ParallelDeferred) {
  test_pinned_resource upstream;
  CUDA_CALL(cudaFree(nullptr));  // initialize device context
  {
    using pool_t = deferred_dealloc_pool<memory_kind::pinned, any_context,
                                        coalescing_free_tree, spinlock>;
    pool_t pool(&upstream);
    std::vector<std::thread> threads;

    std::vector<block> blocks;
    spinlock mtx;

    for (int i = 0; i < 4; i++) {
      threads.emplace_back([&]() {
        PoolTest(pool, blocks, mtx, 50000);
      });
    }

    for (auto &t : threads)
      t.join();

    for (auto &blk : blocks)
      pool.deallocate(blk.ptr, blk.size);
  }
  upstream.check_leaks();
}

}  // namespace test
}  // namespace mm
}  // namespace dali
