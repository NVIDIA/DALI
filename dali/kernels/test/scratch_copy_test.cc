// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <numeric>
#include "dali/kernels/kernel_req.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/scratch_copy_impl.h"
#include "dali/core/tuple_helpers.h"

namespace dali {
namespace kernels {

template <typename C, typename = element_t<C>>
constexpr auto size_bytes(const C &c) { return size(c) * sizeof(element_t<C>); }

TEST(Scratchpad, ToContiguous) {
  ScratchpadEstimator se;
  std::vector<char> chars(301);
  std::vector<float> floats(1927);
  std::vector<int16_t> shorts(1233);
  std::iota(chars.begin(), chars.end(), 1);
  std::iota(floats.begin(), floats.end(), 1);
  std::iota(shorts.begin(), shorts.end(), 1);

  se.add<mm::memory_kind::device, char>(chars.size());
  se.add<mm::memory_kind::device, float>(floats.size());
  se.add<mm::memory_kind::device, int16_t>(shorts.size());
  se.add<mm::memory_kind::host, char>(chars.size());
  se.add<mm::memory_kind::host, float>(floats.size());
  se.add<mm::memory_kind::host, int16_t>(shorts.size());
  ScratchpadAllocator sa;
  sa.Reserve(se.sizes);
  auto scratchpad = sa.GetScratchpad();
  auto ptrs = ToContiguousGPUMem(scratchpad, 0, chars, floats, shorts);
  std::vector<char> chars2(chars.size());
  std::vector<float> floats2(floats.size());
  std::vector<int16_t> shorts2(shorts.size());
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(
    cudaMemcpy(chars2.data(), std::get<0>(ptrs), size_bytes(chars2), cudaMemcpyDeviceToHost));
  CUDA_CALL(
    cudaMemcpy(floats2.data(), std::get<1>(ptrs), size_bytes(floats2), cudaMemcpyDeviceToHost));
  CUDA_CALL(
    cudaMemcpy(shorts2.data(), std::get<2>(ptrs), size_bytes(shorts2), cudaMemcpyDeviceToHost));

  EXPECT_EQ(chars, chars);
  EXPECT_EQ(floats, floats2);
  EXPECT_EQ(shorts, shorts2);

  ptrs = ToContiguousHostMem(scratchpad, chars, floats, shorts);
  size_t offsets[4] = { 0xdead, 0xdead, 0xdead, 0xdead };
  memcpy(chars2.data(), std::get<0>(ptrs), size_bytes(chars2));
  memcpy(floats2.data(), std::get<1>(ptrs), size_bytes(floats2));
  memcpy(shorts2.data(), std::get<2>(ptrs), size_bytes(shorts2));

  EXPECT_EQ(chars, chars);
  EXPECT_EQ(floats, floats2);
  EXPECT_EQ(shorts, shorts2);
}

}  // namespace kernels
}  // namespace dali
