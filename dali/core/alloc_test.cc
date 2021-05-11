// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/core/mm/memory.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/mm/mm_test_utils.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/mm/detail/align.h"

namespace dali {

TEST(Alloc, Host) {
  mm::test::test_host_resource mr;
  mm::uptr<int> data = mm::alloc_raw_unique<int>(&mr, 3);
  memset(data.get(), 0x55, sizeof(int) * 3);
  EXPECT_NO_THROW(data.reset());
  EXPECT_NO_THROW(mr.reset());
}

TEST(Alloc, HostOveraligned) {
  mm::test::test_host_resource mr;
  mm::uptr<int> data = mm::alloc_raw_unique<int>(&mr, 3, 0x10000);
  EXPECT_TRUE(mm::detail::is_aligned(data.get(), 0x10000));
  memset(data.get(), 0x55, sizeof(int) * 3);
  EXPECT_NO_THROW(data.reset());
  EXPECT_NO_THROW(mr.reset());
}

TEST(Alloc, AsyncDev) {
  mm::test::test_dev_pool_resource mr;
  CUDAStream stream = CUDAStream::Create(true);
  mm::async_uptr<int> data = mm::alloc_raw_async_unique<int>(&mr, 1000, stream, stream);
  CUDA_CALL(cudaMemsetAsync(data.get(), 0xff, 1000*sizeof(int), stream));
  int *ptr = data.get();
  data.reset();
  mm::async_uptr<int> data2 = mm::alloc_raw_async_unique<int>(&mr, 1000, stream, mm::host_sync);
  EXPECT_EQ(data2.get(), ptr);
  data2.reset();
  CUDA_CALL(cudaStreamSynchronize(stream));
  EXPECT_NO_THROW(mr.reset());
}

TEST(Alloc, AsyncDevDefault) {
  CUDAStream stream = CUDAStream::Create(true);
  mm::async_uptr<int> data = mm::alloc_raw_async_unique<int, mm::memory_kind::device>(
        1000, stream, stream);
  CUDA_CALL(cudaMemsetAsync(data.get(), 0xff, 1000*sizeof(int), stream));
  data.reset();
  CUDA_CALL(cudaStreamSynchronize(stream));
}

TEST(Alloc, AsyncPinnedDefault) {
  CUDAStream stream = CUDAStream::Create(true);
  mm::async_uptr<int> data = mm::alloc_raw_async_unique<int, mm::memory_kind::pinned>(
        1000, nullptr, stream);
  memset(data.get(), 0x55, sizeof(int) * 3);
  data.reset();
  CUDA_CALL(cudaStreamSynchronize(stream));
}


TEST(Alloc, HostSharedDefault) {
  mm::test::test_host_resource mr;
  std::shared_ptr<int> data = mm::alloc_raw_shared<int, mm::memory_kind::host>(3);
  memset(data.get(), 0x55, sizeof(int) * 3);
  EXPECT_NO_THROW(data.reset());
  EXPECT_NO_THROW(mr.reset());
}

TEST(Alloc, AsyncDevShared) {
  mm::test::test_dev_pool_resource mr;
  CUDAStream stream = CUDAStream::Create(true);
  std::shared_ptr<int> data = mm::alloc_raw_async_shared<int>(&mr, 1000, stream, stream);
  CUDA_CALL(cudaMemsetAsync(data.get(), 0xff, 1000*sizeof(int), stream));
  int *ptr = data.get();
  data.reset();
  std::shared_ptr<int> data2 = mm::alloc_raw_async_shared<int>(&mr, 1000, stream, mm::host_sync);
  EXPECT_EQ(data2.get(), ptr);
  data2.reset();
  CUDA_CALL(cudaStreamSynchronize(stream));
  EXPECT_NO_THROW(mr.reset());
}

TEST(Alloc, AsyncDevDefaultShared) {
  CUDAStream stream = CUDAStream::Create(true);
  std::shared_ptr<int> data = mm::alloc_raw_async_shared<int, mm::memory_kind::device>(
        1000, stream, stream);
  CUDA_CALL(cudaMemsetAsync(data.get(), 0xff, 1000*sizeof(int), stream));
  data.reset();
  CUDA_CALL(cudaStreamSynchronize(stream));
}

TEST(Alloc, AsyncPinnedDefaultShared) {
  CUDAStream stream = CUDAStream::Create(true);
  std::shared_ptr<int> data = mm::alloc_raw_async_shared<int, mm::memory_kind::pinned>(
        1000, nullptr, stream);
  memset(data.get(), 0x55, sizeof(int) * 3);
  data.reset();
  CUDA_CALL(cudaStreamSynchronize(stream));
}


}  // namespace dali
