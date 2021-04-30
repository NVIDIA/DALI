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

namespace dali {

TEST(Alloc, Host) {
  mm::test::test_host_resource mr;
  auto data = mm::alloc_raw<int>(&mr, 3);
  memset(data.get(), 0x55, sizeof(int) * 3);
  EXPECT_NO_THROW(data.reset());
  EXPECT_NO_THROW(mr.reset());
}

TEST(Alloc, AsyncDev) {
  mm::test::test_dev_pool_resource mr;
  CUDAStream stream = CUDAStream::Create(true);
  auto data = mm::alloc_raw_async<int>(&mr, 1000, stream, stream);
  cudaMemsetAsync(data.get(), 0xff, 1000*sizeof(int), stream);
  int *ptr = data.get();
  data.reset();
  auto data2 = mm::alloc_raw_async<int>(&mr, 1000, stream, mm::host_sync);
  EXPECT_EQ(data2.get(), ptr);
  data2.reset();
  EXPECT_NO_THROW(mr.reset());
}

TEST(Alloc, AsyncDevDefault) {
  CUDAStream stream = CUDAStream::Create(true);
  auto data = mm::alloc_raw_async<int, mm::memory_kind::device>(1000, stream, stream);
  cudaMemsetAsync(data.get(), 0xff, 1000*sizeof(int), stream);

  int *ptr = data.get();
  data.reset();
  auto data2 = mm::alloc_raw_async<int, mm::memory_kind::device>(1000, stream, mm::host_sync);
  EXPECT_EQ(data2.get(), ptr);
  data2.reset();
}

}  // namespace dali
