// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <cstring>
#include "dali/kernels/alloc.h"

namespace dali {
namespace kernels {

TEST(KernelAlloc, AllocFree) {
  size_t size = 1<<20;  // 1 MiB
  const char *names[static_cast<int>(AllocType::Count)] = { "Host", "Pinned", "GPU", "Unified" };
  for (int i = 0; i < static_cast<int>(AllocType::Count); i++) {
    AllocType alloc = static_cast<AllocType>(i);
    void *mem = memory::Allocate(alloc, size);
    EXPECT_EQ(cudaGetLastError(), 0) << "Error when allocating for " << names[i];
    ASSERT_NE(mem, nullptr);
    memory::GetDeleter(alloc)(mem);
    EXPECT_EQ(cudaGetLastError(), 0) << "Error when freeing for " << names[i];
  }
}

TEST(KernelAlloc, HostDevice) {
  (void)cudaGetLastError();
  size_t size = 1<<20;  // 1 MiB
  void *pinned = memory::Allocate(AllocType::Pinned, size);
  void *plain = memory::Allocate(AllocType::Host, size);
  void *gpu = memory::Allocate(AllocType::GPU, size);

  ASSERT_NE(pinned, nullptr);
  ASSERT_NE(plain, nullptr);
  ASSERT_NE(gpu, nullptr);

  int *data = reinterpret_cast<int*>(pinned);
  for (size_t i = 0; i < size/sizeof(int); i++)
    data[i] = i*i + 5;
  std::memset(plain, 0, size);

  EXPECT_EQ(cudaMemcpy(gpu, pinned, size, cudaMemcpyHostToDevice), 0);
  EXPECT_EQ(cudaMemcpy(plain, gpu, size, cudaMemcpyDeviceToHost), 0);

  EXPECT_EQ(std::memcmp(plain, pinned, size), 0);

  auto pinned_deallocator = memory::GetDeleter(AllocType::Pinned);
  auto host_deallocator = memory::GetDeleter(AllocType::Host);
  auto gpu_deallocator = memory::GetDeleter(AllocType::GPU);

  int count = 1;
  cudaGetDeviceCount(&count);
  if (count > 1)
    cudaSetDevice(1);
  EXPECT_EQ(cudaGetLastError(), 0);
  pinned_deallocator(pinned);
  host_deallocator(plain);
  gpu_deallocator(gpu);
  EXPECT_EQ(cudaGetLastError(), 0);
  cudaSetDevice(0);
}

TEST(KernelAlloc, Unique) {
  size_t size = 1<<20;  // 1 M
  const char *names[static_cast<int>(AllocType::Count)] = { "Host", "Pinned", "GPU", "Unified" };
  for (int i = 0; i < static_cast<int>(AllocType::Count); i++) {
    AllocType alloc = static_cast<AllocType>(i);
    auto ptr = memory::alloc_unique<float>(alloc, size);
    EXPECT_EQ(cudaGetLastError(), 0) << "Error when allocating for " << names[i];
    ASSERT_NE(ptr, nullptr);
    ptr.reset();
    EXPECT_EQ(cudaGetLastError(), 0) << "Error when freeing for " << names[i];
  }
}

TEST(KernelAlloc, Shared) {
  size_t size = 1<<20;  // 1 M
  const char *names[static_cast<int>(AllocType::Count)] = { "Host", "Pinned", "GPU", "Unified" };
  for (int i = 0; i < static_cast<int>(AllocType::Count); i++) {
    AllocType alloc = static_cast<AllocType>(i);
    std::shared_ptr<float> ptr = memory::alloc_shared<float>(alloc, size);
    EXPECT_EQ(cudaGetLastError(), 0) << "Error when allocating for " << names[i];
    std::shared_ptr<float> ptr2 = ptr;
    ASSERT_NE(ptr, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ptr.reset();
    ptr2.reset();
    EXPECT_EQ(cudaGetLastError(), 0) << "Error when freeing for " << names[i];
  }
}


}  // namespace kernels
}  // namespace dali
