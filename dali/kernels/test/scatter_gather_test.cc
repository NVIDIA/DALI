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
#include <vector>
#include <algorithm>
#include "dali/kernels/common/scatter_gather.h"

namespace dali {
namespace kernels {

namespace detail {
inline bool operator==(const dali::kernels::detail::CopyRange &a,
                       const dali::kernels::detail::CopyRange &b) {
  return a.src == b.src && a.dst == b.dst && a.size == b.size;
}
}  // namespace detail

TEST(ScatterGather, Coalesce) {
  static char A1[1<<12];
  static char A2[1<<12];
  using detail::CopyRange;
  std::vector<CopyRange> in;
  std::vector<CopyRange> ref;
  in.push_back({ A1 + 1000, A2 + 2000, 10 });
  in.push_back({ A1 + 0, A2 + 0, 120 });
  in.push_back({ A1 + 120, A2 + 120, 10 });
  in.push_back({ A1 + 130, A2 + 130, 20 });
  in.push_back({ A1 + 150, A2 + 160, 10 });
  in.push_back({ A1 + 170, A2 + 170, 10 });
  in.push_back({ A1 + 180, A2 + 180, 30 });

  ref.push_back({ A1 + 0, A2 + 0, 150 });
  ref.push_back({ A1 + 150, A2 + 160, 10 });
  ref.push_back({ A1 + 170, A2 + 170, 40 });
  ref.push_back({ A1 + 1000, A2 + 2000, 10 });

  size_t n = detail::Coalesce(make_span(in.data(), in.size()));
  ASSERT_LE(n, in.size());
  EXPECT_EQ(n, ref.size());

  in.resize(n);
  EXPECT_EQ(in, ref);
}

TEST(ScatterGather, Copy) {
  const size_t max_l = 1024;
  std::vector<char> in(1<<20);
  std::vector<char> out(1<<20);
  unsigned seed = 42;

  auto in_ptr = kernels::memory::alloc_unique<char>(AllocType::GPU, in.size());
  auto out_ptr = kernels::memory::alloc_unique<char>(AllocType::GPU, out.size());

  std::vector<detail::CopyRange> ranges;
  std::vector<detail::CopyRange> back_ranges;

  size_t i = 0, j = 0;
  for (;;) {
    detail::CopyRange r;

    if ((rand_r(&seed)&3) == 0) {
      i += rand_r(&seed) % max_l;
      j += rand_r(&seed) % max_l;
    }

    size_t l = rand_r(&seed) % max_l + 1;
    if (i + l > in.size() || j + l > out.size())
      break;

    for (size_t x = i; x < i + l; x++)
        in[x] = rand_r(&seed);


    r.src = in_ptr.get() + i;
    r.dst = out_ptr.get() + j;
    r.size = l;
    ranges.push_back(r);
    r.dst = in_ptr.get() + i;
    r.src = out_ptr.get() + j;
    back_ranges.push_back(r);

    i += l;
    j += l;
  }

  std::random_shuffle(ranges.begin(), ranges.end());
  std::random_shuffle(back_ranges.begin(), back_ranges.end());

  cudaMemcpy(in_ptr.get(), in.data(), in.size(), cudaMemcpyHostToDevice);
  cudaMemset(out_ptr.get(), 0, out.size());

  ScatterGatherGPU sg(64);
  // copy
  for (auto &r : ranges)
    sg.AddCopy(r.dst, r.src, r.size);
  sg.Run(0, true, false);  // use Kernel

  // copy back
  cudaMemset(in_ptr.get(), 0, in.size());
  for (auto &r : back_ranges)
    sg.AddCopy(r.dst, r.src, r.size);
  sg.Run(0, true, true);  // use cudaMemcpyAsync
  cudaMemcpy(out.data(), in_ptr.get(), in.size(), cudaMemcpyDeviceToHost);

  EXPECT_EQ(in, out);
}

}  // namespace kernels
}  // namespace dali
