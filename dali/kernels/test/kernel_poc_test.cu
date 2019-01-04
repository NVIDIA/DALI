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
#include <random>
#include <vector>
#include "dali/kernels/kernel.h"
#include "dali/kernels/test/test_tensors.h"
#include "dali/kernels/tensor_shape_str.h"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/test/kernel_test_utils.h"

using std::cout;
using std::cerr;
using std::endl;

namespace dali {
namespace kernels {

template <typename Input1, typename Input2, typename Output>
__global__ void
ElementwiseMAD(size_t n, Output *o, const Input1 *i1, const Input2 *i2, float alpha) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    o[idx] = i1[idx] * alpha + i2[idx];
}

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

    cout << o.data << " " << i1.data << " " << i2.data << " n = " << n << endl;

    ElementwiseMAD<<<grid, block, 0, context.gpu.stream>>>(n, o.data, i1.data, i2.data, A);

    cout << "Kernel done" << endl;
  }
};

template <typename Kernel_>
class KernelPoC_GPU : public ::testing::Test, public testing::SimpleKernelTestBase<Kernel_> {
};


using PoC_MAD_GPU = ::testing::Types<
  MADKernelGPU<float, float, float>,
  MADKernelGPU<int,   float, float>,
  MADKernelGPU<float, int,   float>,
  MADKernelGPU<int,   int,   int>
>;

TYPED_TEST_CASE(KernelPoC_GPU, PoC_MAD_GPU);

TYPED_TEST(KernelPoC_GPU, All) {
  using MyType = typename std::remove_pointer<decltype(this)>::type;
  using Input1 = typename MyType::template InputElement<0>;
  using Input2 = typename MyType::template InputElement<1>;
  using Output = typename MyType::template OutputElement<0>;
  typename MyType::Kernel K;

  std::mt19937_64 rng;

  InListGPU<Input1, 3> i1;
  InListGPU<Input2, 3> i2;
  OutListGPU<Output, 3> o1, o2;

  KernelContext ctx;

  TestTensorList<Input1> tl1;
  TestTensorList<Input2> tl2;
  TestTensorList<Output> tlo1, tlo2, tlref;

  std::vector<TensorShape<3>> shapes;
  auto size_dist = uniform_distribution(128, 256);
  for (int i = 0; i < 3; i++)
    shapes.push_back({ i+1, size_dist(rng), size_dist(rng)});
  TensorListShape<3> list_shape(shapes);

  tl1.reshape(list_shape);
  tl2.reshape(list_shape);

  Input1 max1 = std::is_integral<Input1>::value ? 100 : 1;
  Input2 max2 = std::is_integral<Input2>::value ? 100 : 1;
  UniformRandomFill(tl1.template cpu<3>(0), rng, 0, max1);
  UniformRandomFill(tl2.template cpu<3>(0), rng, 0, max2);

  i1 = tl1.template gpu<3>();
  i2 = tl2.template gpu<3>();
  auto i1_cpu = tl1.template cpu<3>();
  auto i2_cpu = tl2.template cpu<3>();

  float a = 0.5f;

  auto req = K.GetRequirements(ctx, i1, i2, a);
  ASSERT_EQ((int)req.output_shapes.size(), 1);
  ASSERT_NO_FATAL_FAILURE(CheckEqual(req.output_shapes[0], i1.shape));

  // Kernel's native Run
  tlo1.reshape(req.output_shapes[0]);
  o1 = tlo1.template gpu<3>();
  K.Run(ctx, o1, i1, i2, a);

  // use uniform call with argument tuples
  tlo2.reshape(req.output_shapes[0]);
  o2 = tlo2.template gpu<3>();
  kernels::kernel::Run<decltype(K)>(ctx, std::tie(o2), std::tie(i1, i2), std::make_tuple(a) );

  // verify that shape hasn't changed
  ASSERT_NO_FATAL_FAILURE(CheckEqual(o1.shape, i1.shape));
  ASSERT_NO_FATAL_FAILURE(CheckEqual(o2.shape, i1.shape));

  tlref.reshape(list_shape);
  auto ref = tlref.template cpu<3>(0);
  ptrdiff_t total = ref.num_elements();

  // calculate the reference - since it's purely elementwise,
  // we can skip the tedious multidimensional indexing
  for (ptrdiff_t i = 0; i < total; i++) {
    ref.data[i] = i1_cpu.data[i] * a + i2_cpu.data[i];
  }

  auto o1_cpu = tlo1.template cpu<3>();
  auto o2_cpu = tlo2.template cpu<3>();

  Check(o1_cpu, ref, EqualEps(1e-6));

  // native and uniform calls should yield bit-exact results
  Check(o1_cpu, o2_cpu);
}

}  // namespace kernels
}  // namespace dali
