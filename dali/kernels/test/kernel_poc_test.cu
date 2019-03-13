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
#include "dali/kernels/test/kernel_poc_test.h"

namespace dali {
namespace kernels {

// Performs elementwise MAD (multiply-add).
template <typename Input1, typename Input2, typename Output>
__global__ void
ElementwiseMAD(size_t n, Output *o, const Input1 *i1, const Input2 *i2, float alpha) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    o[idx] = i1[idx] * alpha + i2[idx];
}

// Performs elementwise MAD (multiply-add).
template <typename Input1, typename Input2, typename Output>
struct MADKernelGPU {
  static KernelRequirements GetRequirements(
      KernelContext &context,
      const InListGPU<Input1, 3> &i1,
      const InListGPU<Input2, 3> &i2,
      float A) {
    KernelRequirements req;
    req.output_shapes = { i1.shape };
    return req;
  }

  static void Run(
      KernelContext &context,
      const OutListGPU<Output, 3> &o,
      const InListGPU<Input1, 3> &i1,
      const InListGPU<Input2, 3> &i2,
      float A) {
    auto n = i1.num_elements();
    assert(i2.num_elements() == n);
    assert(o.num_elements() == n);
    size_t block = 1024;
    size_t grid = (n + block - 1) / block;

    ElementwiseMAD<<<grid, block, 0, context.gpu.stream>>>(n, o.data, i1.data, i2.data, A);
  }
};

template <typename Kernel_>
class KernelPoC_GPU : public ::testing::Test, public KernelPoCFixture<StorageGPU, Kernel_> {
};


using PoC_MAD_GPU = ::testing::Types<
  MADKernelGPU<float, float, float>,
  MADKernelGPU<int,   float, float>,
  MADKernelGPU<float, int,   float>,
  MADKernelGPU<int,   int,   int>
>;

TYPED_TEST_SUITE(KernelPoC_GPU, PoC_MAD_GPU);

TYPED_TEST(KernelPoC_GPU, All) {
  this->RunImpl();
}

}  // namespace kernels
}  // namespace dali
