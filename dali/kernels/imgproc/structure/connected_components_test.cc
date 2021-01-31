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
#include <cassert>
#include <vector>
#include "dali/kernels/imgproc/structure/connected_components.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {

TEST(ConnectedComponents, Compact) {
  std::vector<int> labels = {
     0,  0,  0,  0,  4,  4,  4,  4,
     0,  0, 10,  0,  4,  4, 10,  4,
    -1,  0, 10, 10, 10, 10, 10,  4,
    -1, -1, -1, -1, -1, 10, -1,  4,
    32, 32, 10, 10, 10, 10,  4,  4,
  };

  std::vector<int> compacted = {
     1,  1,  1,  1,  2,  2,  2,  2,
     1,  1,  3,  1,  2,  2,  3,  2,
     0,  1,  3,  3,  3,  3,  3,  2,
     0,  0,  0,  0,  0,  3,  0,  2,
     4,  4,  3,  3,  3,  3,  2,  2,
  };

  int64_t n = connected_components::detail::CompactLabels(labels.data(), labels.size(), 0);
  EXPECT_EQ(n, 4);
  EXPECT_EQ(labels, compacted);
}

TEST(ConnectedComponets, 1D) {
  const int N = 16;
  int input[N]  = { 0, 0, 1, 1, 2, 1, 2, 0, 0, 2, 2, 2, 0, 0, 1, 1 };
  int output[N] = {};
  int ref[N]    = { 0, 0, 1, 1, 2, 3, 4, 0, 0, 5, 5, 5, 0, 0, 6, 6 };
  InTensorCPU<int, 1> in = make_tensor_cpu<1>(input, { N });
  OutTensorCPU<int, 1> out = make_tensor_cpu<1>(output, { N });
  connected_components::LabelConnectedRegions(out, in);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(output[i], ref[i]) << " at index " << i;
  }
}

TEST(ConnectedComponets, 2D) {
  const int H = 5;
  const int W = 8;

  const int objects[H*W] = {
     5,  5,  5,  5,  2,  2,  2,  2,
     5,  5,  3,  5,  2,  2,  3,  2,
    -2,  5,  3,  3,  3,  3,  3,  2,
    -2, -2, -2, -2, -2,  3, -2,  2,
     5,  5,  3,  3,  3,  3,  2, -2,
  };

  const unsigned labels[H*W] = {
     1,  1,  1,  1,  2,  2,  2,  2,
     1,  1,  3,  1,  2,  2,  3,  2,
     0,  1,  3,  3,  3,  3,  3,  2,
     0,  0,  0,  0,  0,  3,  0,  2,
     4,  4,  3,  3,  3,  3,  5,  0,
  };

  unsigned output[H*W] = {};

  InTensorCPU<int, 2> in = make_tensor_cpu<2>(objects, { H, W });
  OutTensorCPU<unsigned, 2> out = make_tensor_cpu<2>(output, { H, W });
  InTensorCPU<unsigned, 2> ref = make_tensor_cpu<2>(labels, { H, W });
  int64_t n = connected_components::LabelConnectedRegions(out, in, 0, -2);
  EXPECT_EQ(n, 5);
  Check(out, ref);
}

TEST(ConnectedComponets, 2D_WithDegenerateDims) {
  const int H = 5;
  const int W = 8;

  const int objects[H*W] = {
     5,  5,  5,  5,  2,  2,  2,  2,
     5,  5,  3,  5,  2,  2,  3,  2,
    -2,  5,  3,  3,  3,  3,  3,  2,
    -2, -2, -2, -2, -2,  3, -2,  2,
     5,  5,  3,  3,  3,  3,  2, -2,
  };

  const unsigned labels[H*W] = {
     1,  1,  1,  1,  2,  2,  2,  2,
     1,  1,  3,  1,  2,  2,  3,  2,
     0,  1,  3,  3,  3,  3,  3,  2,
     0,  0,  0,  0,  0,  3,  0,  2,
     4,  4,  3,  3,  3,  3,  5,  0,
  };

  unsigned output[H*W] = {};

  InTensorCPU<int, 4> in = make_tensor_cpu<4>(objects, { H, 1, W, 1 });
  OutTensorCPU<unsigned, 4> out = make_tensor_cpu<4>(output, { H, 1, W, 1 });
  InTensorCPU<unsigned, 4> ref = make_tensor_cpu<4>(labels, { H, 1, W, 1 });
  int64_t n = connected_components::LabelConnectedRegions(out, in, 0, -2);
  EXPECT_EQ(n, 5);
  Check(out, ref);
}


TEST(ConnectedComponets, 3D) {
  const int D = 2;
  const int H = 4;
  const int W = 6;

  const char objects[D*H*W+1] =
    "ABB CA"
    " BA AA"
    "B A   "
    "    DA"

    "AAA CC"
    " BA CA"
    "BB   A"
    "    DA";

  const int labels[D*H*W] = {
     0,  1,  1, -1,  2,  3,
    -1,  1,  0, -1,  3,  3,
     1, -1,  0, -1, -1, -1,
    -1, -1, -1, -1,  4,  3,

     0,  0,  0, -1,  2,  2,
    -1,  1,  0, -1,  2,  3,
     1,  1, -1, -1, -1,  3,
    -1, -1, -1, -1,  4,  3,
  };

  int output[D*H*W] = {};

  InTensorCPU<char, 3> in = make_tensor_cpu<3>(objects, { D, H, W });
  OutTensorCPU<int, 3> out = make_tensor_cpu<3>(output, { D, H, W });
  InTensorCPU<int, 3> ref = make_tensor_cpu<3>(labels, { D, H, W });
  int64_t n = connected_components::LabelConnectedRegions(out, in, -1, ' ');
  EXPECT_EQ(n, 5);
  Check(out, ref);
}

}  // namespace kernels
}  // namespace dali
