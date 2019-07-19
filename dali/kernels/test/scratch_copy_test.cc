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
  std::iota(chars.begin(), chars.end(), 1);
  std::iota(floats.begin(), floats.end(), 1);

  se.add<char>(AllocType::GPU, chars.size());
  se.add<float>(AllocType::GPU, floats.size());
  se.add<char>(AllocType::Host, chars.size());
  se.add<float>(AllocType::Host, floats.size());
  ScratchpadAllocator sa;
  sa.Reserve(se.sizes);
  auto scratchpad = sa.GetScratchpad();
  auto ptrs = ToContiguousGPUMem(scratchpad, 0, chars, floats);
  std::vector<char> chars2(chars.size());
  std::vector<float> floats2(floats.size());
  cudaDeviceSynchronize();
  cudaMemcpy(chars2.data(), std::get<0>(ptrs), size_bytes(chars2), cudaMemcpyDeviceToHost);
  cudaMemcpy(floats2.data(), std::get<1>(ptrs), size_bytes(floats2), cudaMemcpyDeviceToHost);

  EXPECT_EQ(chars, chars);
  EXPECT_EQ(floats, floats2);

  ptrs = ToContiguousHostMem(scratchpad, chars, floats);
  memcpy(chars2.data(), std::get<0>(ptrs), size_bytes(chars2));
  memcpy(floats2.data(), std::get<1>(ptrs), size_bytes(floats2));

  EXPECT_EQ(chars, chars);
  EXPECT_EQ(floats, floats2);
}

}  // namespace kernels
}  // namespace dali
