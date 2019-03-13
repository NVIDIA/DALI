// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/test/kernel_poc_test.h"

using std::cout;
using std::cerr;
using std::endl;

namespace dali {
namespace kernels {

// Performs elementwise MAD (multiply-add).
template <typename Input1, typename Input2, typename Output>
struct MADKernel {
  static KernelRequirements GetRequirements(
      KernelContext &context,
      const InListCPU<Input1, 3> &i1,
      const InListCPU<Input2, 3> &i2,
      float A) {
    KernelRequirements req;
    req.output_shapes = { i1.shape };
    return req;
  }

  static void Run(
      KernelContext &context,
      const OutListCPU<Output, 3> &o,
      const InListCPU<Input1, 3> &i1,
      const InListCPU<Input2, 3> &i2,
      float A) {
    int samples = i1.num_samples();
    for (int n = 0; n < samples; n++) {
      auto tshape = i1.shape.tensor_shape_span(n);
      int d = tshape[0];
      int h = tshape[1];
      int w = tshape[2];

      auto t1 = i1[n];
      auto t2 = i2[n];
      auto to = o[n];

      for (int z = 0; z < d; z++) {
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            *to(z, y, x) = *t1(z, y, x) * A + *t2(z, y, x);
          }
        }
      }
    }
  }
};

template <typename Kernel_>
class KernelPoC_CPU : public ::testing::Test, public KernelPoCFixture<StorageCPU, Kernel_> {
};

using PoC_MAD = ::testing::Types<
  MADKernel<float, float, float>,
  MADKernel<int,   float, float>,
  MADKernel<float, int,   float>,
  MADKernel<int,   int,   int>
>;

TYPED_TEST_SUITE(KernelPoC_CPU, PoC_MAD);

TYPED_TEST(KernelPoC_CPU, All) {
  this->RunImpl();
}

}  // namespace kernels
}  // namespace dali
