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

#include <gtest/gtest.h>
#include <utility>
#include "dali/operators/reader/gds_mem.h"
#include "dali/core/cuda_stream.h"

namespace dali {
namespace gds {

TEST(GDSMem, AllocatorMultiDevice) {
  int ndev;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    GTEST_SKIP() << "This test requires more than one CUDA capable device to run.";
    return;
  }
  ASSERT_NE(GDSAllocator::get(0), GDSAllocator::get(1));
}

TEST(GDSMem, Allocator) {
  auto alloc = GDSAllocator::get();
  auto unq = alloc->alloc_unique(1024);
  ASSERT_NE(unq, nullptr);
  CUDA_CALL(cudaMemset(unq.get(), 0, 1024));
  unq.reset();
  alloc.reset();
}

TEST(GDSMem, StagingEngine) {
  CUDAStream stream = CUDAStream::Create(true);
  GDSStagingEngine engn;
  size_t chunk = engn.chunk_size();
  auto target_buf = mm::alloc_raw_unique<uint8_t, mm::memory_kind::device>(chunk * 2);
  CUDA_CALL(cudaMemset(target_buf.get(), 0xcc, chunk * 2));
  CUDA_CALL(cudaDeviceSynchronize());
  engn.set_stream(stream);
  auto buf0 = engn.get_staging_buffer();
  auto buf1 = engn.get_staging_buffer();
  engn.return_unused(std::move(buf0));
  auto buf2 = engn.get_staging_buffer(buf1.at(chunk));
  CUDA_CALL(cudaMemsetAsync(buf1.at(0), 0xaa, chunk, stream));
  CUDA_CALL(cudaMemsetAsync(buf2.at(0), 0xbb, chunk, stream));
  engn.copy_to_client(target_buf.get(), chunk, std::move(buf1), 0);
  engn.copy_to_client(target_buf.get() + chunk, chunk, std::move(buf2), 0);
  engn.commit();
  std::vector<uint8_t> host(chunk * 2);
  CUDA_CALL(cudaMemcpyAsync(host.data(), target_buf.get(), 2 * chunk,
                            cudaMemcpyDeviceToHost, stream));
  for (size_t i = 0; i < chunk; i++)
    ASSERT_EQ(host[i], 0xaa);
  for (size_t i = chunk; i < 2*chunk; i++)
    ASSERT_EQ(host[i], 0xbb);
}

}  // namespace gds
}  // namespace dali
