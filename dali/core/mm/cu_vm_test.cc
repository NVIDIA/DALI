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

#include "dali/core/mm/cu_vm.h"
#include <gtest/gtest.h>
#include <chrono>
#include <vector>

#if DALI_USE_CUDA_VM_MAP

namespace dali {
namespace mm {
namespace test {

TEST(CUMemAddressRange, Reserve) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory API not supported on this machine";

  int64_t requested = 4000000;
  cuvm::CUMemAddressRange range = cuvm::CUMemAddressRange::Reserve(requested);
  EXPECT_GE(static_cast<int64_t>(range.size()), requested);
  CUpointer_attribute attrs[3] = {
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE,
    CU_POINTER_ATTRIBUTE_MAPPED
  };
  CUdeviceptr start = 0;
  size_t size = 0;
  bool mapped = false;
  void *data[3] = { &start, &size, &mapped };
  CUDA_CALL(cuPointerGetAttributes(3, attrs, data, range.ptr() + 100));
  EXPECT_EQ(start, range.ptr());
  EXPECT_EQ(size, range.size());
  EXPECT_FALSE(mapped);
  EXPECT_NO_THROW(range.reset());
}

TEST(CUMemAddressRange, ReserveAndMap) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory API not supported on this machine";

  int64_t virt_size = 10<<20;
  int64_t phys_size = 4<<20;
  cuvm::CUMemAddressRange range = cuvm::CUMemAddressRange::Reserve(virt_size);
  cuvm::CUMem mem = cuvm::CUMem::Create(phys_size);
  CUdeviceptr base = range.ptr();
  EXPECT_GE(static_cast<int64_t>(range.size()), virt_size);
  CUpointer_attribute attrs[3] = {
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE,
    CU_POINTER_ATTRIBUTE_MAPPED
  };
  CUdeviceptr start = 0;
  size_t size = 0;
  bool mapped = false;
  void *data[3] = { &start, &size, &mapped };
  void *ptr = cuvm::Map(base, mem);
  EXPECT_EQ(ptr, reinterpret_cast<void*>(base));
  CUDA_CALL(cuPointerGetAttributes(3, attrs, data, base + 100));
  EXPECT_EQ(start, range.ptr());
  EXPECT_EQ(size, range.size());
  EXPECT_TRUE(mapped);
  EXPECT_NO_THROW(cuvm::Unmap(ptr, mem.size()));
  CUDA_CALL(cuPointerGetAttributes(3, attrs, data, base + 1234));
  EXPECT_EQ(start, range.ptr());
  EXPECT_EQ(size, range.size());
  EXPECT_FALSE(mapped);
  EXPECT_NO_THROW(mem.reset());
  EXPECT_NO_THROW(range.reset());
}


TEST(CUMemAddressRange, ReserveAndMapPiecewise) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory API not supported on this machine";

  int64_t virt_size = 10<<20;
  int64_t phys_size = 4<<20;
  cuvm::CUMemAddressRange range = cuvm::CUMemAddressRange::Reserve(virt_size);
  cuvm::CUMem mem1 = cuvm::CUMem::Create(phys_size);
  cuvm::CUMem mem2 = cuvm::CUMem::Create(phys_size);
  CUdeviceptr base = range.ptr();
  EXPECT_GE(static_cast<int64_t>(range.size()), virt_size);
  CUpointer_attribute attrs[3] = {
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE,
    CU_POINTER_ATTRIBUTE_MAPPED
  };
  CUdeviceptr start = 0;
  size_t size = 0;
  bool mapped = false;
  void *data[3] = { &start, &size, &mapped };
  void *ptr1 = cuvm::Map(base, mem1);
  void *ptr2 = cuvm::Map(base + phys_size, mem2);
  EXPECT_EQ(ptr1, reinterpret_cast<void*>(base));
  EXPECT_EQ(ptr2, reinterpret_cast<void*>(base + phys_size));
  EXPECT_EQ(cudaSuccess, cudaMemset(ptr1, 0, 2*phys_size));
  CUDA_CALL(cuPointerGetAttributes(3, attrs, data, base + 100));
  EXPECT_EQ(start, range.ptr());
  EXPECT_EQ(size, range.size());
  EXPECT_TRUE(mapped);
  CUDA_CALL(cudaDeviceSynchronize());
  EXPECT_NO_THROW(cuvm::Unmap(ptr1, mem1.size()));
  CUDA_CALL(cuPointerGetAttributes(3, attrs, data, base + 1234));
  EXPECT_EQ(start, range.ptr());
  EXPECT_EQ(size, range.size());
  EXPECT_FALSE(mapped);
  EXPECT_NO_THROW(mem1.reset());
  EXPECT_NO_THROW(cuvm::Unmap(ptr2, mem2.size()));
  EXPECT_NO_THROW(mem2.reset());
  EXPECT_NO_THROW(range.reset());
}

template <typename Out = double, typename R, typename P>
inline Out microseconds(std::chrono::duration<R, P> d) {
  return std::chrono::duration_cast<std::chrono::duration<Out, std::micro>>(d).count();
}


TEST(CUMemAddressRange, Perf) {
  if (!cuvm::IsSupported())
    GTEST_SKIP() << "CUDA Virtual Memory API not supported on this machine";

  int64_t virt_size = 10L << 30;
  int64_t phys_size = (64L) << 20;
  std::vector<cuvm::CUMem> phys;
  int N = 16;
  phys.reserve(N);
  auto t0 = std::chrono::high_resolution_clock::now();
  cuvm::CUMemAddressRange range = cuvm::CUMemAddressRange::Reserve(virt_size);
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    phys.push_back(cuvm::CUMem::Create(phys_size));
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  CUdeviceptr p = range.ptr();
  for (int i = 0; i < static_cast<int>(phys.size()); i++) {
    cuvm::Map(p, phys[i]);
    p += phys[i].size();
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  p = range.ptr();
  for (int i = 0; i < static_cast<int>(phys.size()); i++) {
    cuvm::Unmap(p, phys[i].size());
    p += phys[i].size();
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  phys.clear();
  auto t5 = std::chrono::high_resolution_clock::now();
  range.reset();
  auto t6 = std::chrono::high_resolution_clock::now();
  void *mem;
  CUDA_CALL(cudaMalloc(&mem, N*phys_size));
  auto t7 = std::chrono::high_resolution_clock::now();
  CUDA_CALL(cudaFree(mem));
  auto t8 = std::chrono::high_resolution_clock::now();
  std::cout << microseconds(t1 - t0) << std::endl;
  std::cout << microseconds(t2 - t1) << std::endl;
  std::cout << microseconds(t3 - t2) << std::endl;
  std::cout << microseconds(t4 - t3) << std::endl;
  std::cout << microseconds(t5 - t4) << std::endl;
  std::cout << microseconds(t6 - t5) << std::endl;
  std::cout << microseconds(t7 - t6) << std::endl;
  std::cout << microseconds(t8 - t7) << std::endl;
}

}  // namespace test
}  // namespace mm
}  // namespace dali


#endif
