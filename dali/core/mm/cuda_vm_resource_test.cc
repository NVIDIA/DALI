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
#include <vector>
#include "dali/core/random.h"
#include "dali/core/mm/cuda_vm_resource.h"
#include "dali/core/mm/mm_test_utils.h"
#include "dali/core/format.h"

#if DALI_USE_CUDA_VM_MAP

namespace dali {
namespace mm {
namespace test {

class VMResourceTest : public ::testing::Test {
 public:
  void TestAlloc() {
    cuda_vm_resource_base res;
    res.block_size_ = 32 << 20;  // fix the block size at 32 MiB for this test
    const size_t size1 = 100 << 20;  // used for first three allocations
    const size_t size4 = 150 << 20;  // used for the fourth allocation
    void *ptr = res.allocate(size1);
    void *ptr2 = res.allocate(size1);
    EXPECT_EQ(res.va_regions_.size(), 1u);
    EXPECT_EQ(res.va_regions_.front().mapped.find(false), 7);
    res.deallocate(ptr, size1);
    EXPECT_EQ(res.va_regions_.front().available.find(true), 0);
    EXPECT_EQ(res.va_regions_.front().available.find(false), 3);
    void *ptr3 = res.allocate(size1);
    EXPECT_EQ(ptr, ptr3);
    cuda_vm_resource::mem_handle_t blocks[3];
    for (int i = 0; i < 3; i++)
      blocks[i] = res.va_regions_.front().mapping[i];
    res.deallocate(ptr3, size1);
    // let's request more than was deallocated - it should go at the end of VA range
    void *ptr4 = res.allocate(size4);
    EXPECT_EQ(ptr4, static_cast<void*>(static_cast<char*>(ptr2) + size1));
    for (int i = 0; i < 3; i++) {
      // ptr4 should start 200 MiB from start, which is block 6
      // block 7 should be still unmapped, hence i+7.
      // Let's check that the 1st 3 blocks have been reused rather than newly allocated.
      EXPECT_EQ(res.va_regions_.back().mapping[i+7], blocks[i]);
    }
    res.deallocate(ptr2, size1);
    res.deallocate(ptr4, size4);
  }

  void MapRandomBlocks(cuda_vm_resource::va_region &region, int blocks_to_map) {
    assert(blocks_to_map < region.num_blocks());
    std::vector<int> block_idxs(region.num_blocks());
    random_permutation(block_idxs, rng_);
    block_idxs.resize(blocks_to_map);
    for (int blk_idx : block_idxs) {
      region.map_block(blk_idx, cuvm::CUMem::Create(region.block_size));
    }
  }

  // Unlike normal va_region, this one doesn't release the memory blocks upon destruction
  struct va_region_backup : cuda_vm_resource::va_region {
    using va_region::va_region;
    va_region_backup(va_region_backup &&other) = default;

    ~va_region_backup() {
      // suppress purge
      mapping.clear();
    }
  };

  static va_region_backup Backup(const cuda_vm_resource::va_region &in) {
    va_region_backup out(in.address_range, in.block_size);
    out.available_blocks = in.available_blocks;
    out.mapping   = in.mapping;
    out.mapped    = in.mapped;
    out.available = in.available;
    return out;
  }

  void ComparePart(const cuda_vm_resource::va_region &region,
                   int pos,
                   const cuda_vm_resource::va_region &ref) {
    for (int i = 0, j = pos; i < ref.num_blocks(); i++, j++) {
      EXPECT_EQ(region.mapping[j],   ref.mapping[i])   << "@ " << j;
      EXPECT_EQ(region.mapped[j],    ref.mapped[i])    << "@ " << j;
      EXPECT_EQ(region.available[j], ref.available[i]) << "@ " << j;
    }
  }

  void CheckPartEmpty(const cuda_vm_resource::va_region &region, int start, int end) {
    for (int i = start; i < end; i++) {
      EXPECT_EQ(region.mapping[i], CUmemGenericAllocationHandle{});
      EXPECT_FALSE(region.mapped[i]);
      EXPECT_FALSE(region.available[i]);
    }
  }

  void TestRegionExtendAfter() {
    cuda_vm_resource res;
    res.block_size_ = 4 << 20;  // 4 MiB
    const int b1 = 32, b2 = 64;
    const size_t s1 = b1 * res.block_size_;
    const size_t s2 = b2 * res.block_size_;
    res.va_ranges_.push_back(cuvm::CUMemAddressRange::Reserve(s1 + s2));
    cuvm::CUAddressRange total = res.va_ranges_.back();
    cuvm::CUAddressRange part1 = { total.ptr(),           s1 };
    cuvm::CUAddressRange part2 = { total.ptr() + s1,      s2 };
    res.va_add_region(part1);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    MapRandomBlocks(res.va_regions_[0], 10);
    va_region_backup va1 = Backup(res.va_regions_[0]);
    res.va_add_region(part2);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    auto &region = res.va_regions_.back();
    ASSERT_EQ(region.num_blocks(), b1 + b2);
    ComparePart(region, 0, va1);
    CheckPartEmpty(region, b1, b1 + b2);
  }

  void TestRegionExtendBefore() {
    cuda_vm_resource res;
    res.block_size_ = 4 << 20;  // 4 MiB
    const int b1 = 32, b2 = 64;
    const size_t s1 = b1 * res.block_size_;
    const size_t s2 = b2 * res.block_size_;
    res.va_ranges_.push_back(cuvm::CUMemAddressRange::Reserve(s1 + s2));
    cuvm::CUAddressRange total = res.va_ranges_.back();
    cuvm::CUAddressRange part1 = { total.ptr(),           s1 };
    cuvm::CUAddressRange part2 = { total.ptr() + s1,      s2 };
    res.va_add_region(part2);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    MapRandomBlocks(res.va_regions_[0], 10);
    va_region_backup va1 = Backup(res.va_regions_[0]);
    res.va_add_region(part1);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    auto &region = res.va_regions_.back();
    ASSERT_EQ(region.num_blocks(), b1 + b2);
    ComparePart(region, b1, va1);
    CheckPartEmpty(region, 0, b1);
  }

  void TestRegionMerge() {
    cuda_vm_resource res;
    res.block_size_ = 4 << 20;  // 4 MiB
    const int b1 = 32, b2 = 64, b3 = 32;
    const size_t s1 = b1 * res.block_size_;
    const size_t s2 = b2 * res.block_size_;
    const size_t s3 = b3 * res.block_size_;
    res.va_ranges_.push_back(cuvm::CUMemAddressRange::Reserve(s1 + s2 + s3));
    cuvm::CUAddressRange total = res.va_ranges_.back();
    cuvm::CUAddressRange part1 = { total.ptr(),           s1 };
    cuvm::CUAddressRange part2 = { total.ptr() + s1,      s2 };
    cuvm::CUAddressRange part3 = { total.ptr() + s1 + s2, s3 };
    res.va_add_region(part1);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    MapRandomBlocks(res.va_regions_[0], 10);
    va_region_backup va1 = Backup(res.va_regions_[0]);
    res.va_add_region(part3);
    ASSERT_EQ(res.va_regions_.size(), 2u);
    MapRandomBlocks(res.va_regions_[1], 12);
    va_region_backup va3 = Backup(res.va_regions_[1]);
    res.va_add_region(part2);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    auto &region = res.va_regions_.back();
    ASSERT_EQ(region.num_blocks(), b1 + b2 + b3);
    ComparePart(region, 0, va1);
    ComparePart(region, b1 + b2, va3);
    CheckPartEmpty(region, b1, b1 + b2);
  }

  void TestPartialMap() {
    // Test a case when a region to be allocated is partially mapped:
    cuda_vm_resource res;
    size_t block_size = 4 << 20;  // 4 MiB;
    res.block_size_ = block_size;
    void *p1 = res.allocate(block_size);  // allocate one block
    void *p2 = res.allocate(block_size);  // allocate another block
    res.deallocate(p2, block_size);       // now free the second block
    res.flush_deferred();
    res.flush_deferred();
    auto &region = res.va_regions_[0];
    EXPECT_EQ(region.available_blocks, 1);
    EXPECT_EQ(region.mapped.find(false), 2);    // 2 mapped blocks
    EXPECT_EQ(region.available.find(true), 1);  // of which the second is available
    void *p3 = res.allocate(2 * block_size);
    EXPECT_EQ(p2, p3);
    CUDA_CALL(cudaMemset(p1, 0, block_size));
    CUDA_CALL(cudaMemset(p2, 0, 2 * block_size));
    CUDA_CALL(cudaDeviceSynchronize());
    EXPECT_EQ(region.mapped.find(false), 3);
    for (int i = 0; i < 3; i++) {
      EXPECT_NE(region.mapping[i], CUmemGenericAllocationHandle{});
      for (int j = i + 1; j < 3; j++) {
        EXPECT_NE(region.mapping[i], region.mapping[j]);
      }
    }
    res.deallocate(p1, block_size);
    res.deallocate(p3, 2 * block_size);
    res.flush_deferred();
    res.flush_deferred();
    EXPECT_EQ(region.available_blocks, 3);
  }

  std::mt19937_64 rng_{12345};
};

TEST_F(VMResourceTest, BasicTest) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory Management not supported on this platform";
  this->TestAlloc();
}

TEST_F(VMResourceTest, RegionMerge) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory Management not supported on this platform";
  this->TestRegionMerge();
}

TEST_F(VMResourceTest, RegionExtendAfter) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory Management not supported on this platform";
  this->TestRegionExtendAfter();
}

TEST_F(VMResourceTest, RegionExtendBefore) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory Management not supported on this platform";
  this->TestRegionExtendBefore();
}

TEST_F(VMResourceTest, PartialMap) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory Management not supported on this platform";
  this->TestPartialMap();
}

std::string format_size(size_t bytes) {
  std::stringstream ss;
  print(ss, bytes, " bytes");
  if (bytes > (1LLU << 34)) {
    print(ss, " (", bytes >> 30, " GiB)");
  } else if (bytes > (1LLU << 24)) {
    print(ss, " (", bytes >> 20, " MiB)");
  } else if (bytes > (1LLU << 14)) {
    print(ss, " (", bytes >> 10, " KiB)");
  }
  return ss.str();
}

using perfclock = std::chrono::high_resolution_clock;

template <typename Out = double, typename R, typename P>
inline Out microseconds(std::chrono::duration<R, P> d) {
  return std::chrono::duration_cast<std::chrono::duration<Out, std::micro>>(d).count();
}

TEST_F(VMResourceTest, RandomAllocations) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory Management not supported on this platform";

  cuda_vm_resource pool(-1, 4 << 20);  // use small 4 MiB blocks
  std::mt19937_64 rng(12345);
  std::bernoulli_distribution is_free(0.5);
  std::uniform_int_distribution<int> align_dist(0, 8);  // alignment anywhere from 1B to 256B
  std::poisson_distribution<int> size_dist(4 << 20);  // average at 4 MiB
  struct allocation {
    void *ptr;
    size_t size, alignment;
  };
  std::vector<allocation> allocs;

  int num_iter = 100000;

  std::map<uintptr_t, size_t> allocated;

  perfclock::duration alloc_time = {};
  perfclock::duration dealloc_time = {};

  for (int i = 0; i < num_iter; i++) {
    if (is_free(rng) && !allocs.empty()) {
      auto idx = rng() % allocs.size();
      allocation a = allocs[idx];
      auto t0 = perfclock::now();
      pool.deallocate(a.ptr, a.size, a.alignment);
      auto t1 = perfclock::now();
      dealloc_time += t1 - t0;
      std::swap(allocs[idx], allocs.back());
      allocs.pop_back();
    } else {
      allocation a;
      a.size = std::max(1, std::min(size_dist(rng), 256 << 20));  // Limit to 256 MiB
      a.alignment = 1 << align_dist(rng);
      auto t0 = perfclock::now();
      a.ptr = pool.allocate(a.size, a.alignment);
      auto t1 = perfclock::now();
      alloc_time += t1 - t0;
      ASSERT_TRUE(detail::is_aligned(a.ptr, a.alignment));
      CUDA_CALL(cudaMemset(a.ptr, 0, a.size));
      CUDA_CALL(cudaDeviceSynchronize());  // wait now so the deallocation timing is not affected
      allocs.push_back(a);
    }
  }

  for (auto &a : allocs) {
    auto t0 = perfclock::now();
    pool.deallocate(a.ptr, a.size, a.alignment);
    auto t1 = perfclock::now();
    dealloc_time += t1 - t0;
  }
  allocs.clear();

  pool.flush_deferred();
  pool.flush_deferred();

  auto stat = pool.get_stat();
  print(std::cerr,
    "Total allocations:     ", stat.total_allocations, "\n"
    "Peak allocations:      ", stat.peak_allocations, "\n"
    "Average time to allocate:   ", microseconds(alloc_time) / stat.total_allocations, " us\n"
    "Average time to deallocate: ", microseconds(dealloc_time) / stat.total_deallocations, " us\n"
    "Peak allocations size: ", format_size(stat.peak_allocated), "\n\n"
    "Peak allocated blocks: ", stat.peak_allocated_blocks, "\n"
    "Peak allocated physical size: ", format_size(stat.peak_allocated_blocks * pool.block_size()),
    "\n"
    "Unmap operations: ", stat.total_unmaps, "\n"
    "VA size: ", format_size(stat.allocated_va), "\n");

  EXPECT_EQ(stat.curr_allocated, 0u);
  EXPECT_EQ(stat.curr_allocations, 0u);
  EXPECT_EQ(stat.total_allocations, stat.total_deallocations);
}

TEST_F(VMResourceTest, OOM) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory Management not supported on this platform";

  auto hog = []() {
    cuda_vm_resource_base pool(-1, 64 << 20);  // use large 64 MiB blocks
    size_t size = 512_uz << 20;
    while (size) {
      print(std::cerr, "Allocating a block of size ", size, "\n");
      void *ptr = pool.allocate(size);
      pool.deallocate(ptr, size);
      size <<= 1;
    }
  };
  EXPECT_THROW(hog(), std::bad_alloc);
}

}  // namespace test
}  // namespace mm
}  // namespace dali

#endif  // DALI_USE_CUDA_VM_MAP
