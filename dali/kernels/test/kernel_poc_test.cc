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
#include <random>
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
class KernelPoC : public ::testing::Test, public testing::SimpleKernelTestBase<Kernel_> {
};

using PoC_MAD = ::testing::Types<
  MADKernel<float, float, float>,
  MADKernel<int,   float, float>,
  MADKernel<float, int,   float>,
  MADKernel<int,   int,   int>
>;

TYPED_TEST_CASE(KernelPoC, PoC_MAD);

TYPED_TEST(KernelPoC, All) {
  using MyType = typename std::remove_pointer<decltype(this)>::type;
  using Input1 = typename MyType::template InputElement<0>;
  using Input2 = typename MyType::template InputElement<1>;
  using Output = typename MyType::template OutputElement<0>;
  unsigned seed = 123;
  typename MyType::Kernel K;

  InListCPU<Input1, 3> i1;
  InListCPU<Input2, 3> i2;
  OutListCPU<Output, 3> o1, o2;

  KernelContext ctx;

  TestTensorList<Input1> tl1;
  TestTensorList<Input2> tl2;
  TestTensorList<Output> tlo1, tlo2, tlref;

  std::vector<TensorShape<3>> shapes;
  for (int i = 0; i < 3; i++)
    shapes.push_back({ i+1, rand_r(&seed)%128+128, rand_r(&seed)%128+128});
  TensorListShape<3> list_shape(shapes);

  tl1.reshape(list_shape);
  tl2.reshape(list_shape);

  std::mt19937_64 rng;
  Input1 max1 = std::is_integral<Input1>::value ? 100 : 1;
  Input2 max2 = std::is_integral<Input2>::value ? 100 : 1;
  UniformRandomFill(tl1.template cpu<3>(0), rng, 0, max1);
  UniformRandomFill(tl2.template cpu<3>(0), rng, 0, max2);

  i1 = tl1.template cpu<3>();
  i2 = tl2.template cpu<3>();

  float a = 0.5f;

  auto req = K.GetRequirements(ctx, i1, i2, a);
  ASSERT_EQ(req.output_shapes.size(), 1);
  ASSERT_NO_FATAL_FAILURE(CheckEqual(req.output_shapes[0], i1.shape));

  // Kernel's native Run
  tlo1.reshape(req.output_shapes[0]);
  o1 = tlo1.template cpu<3>();
  K.Run(ctx, o1, i1, i2, a);

  // use uniform call with argument tuples
  tlo2.reshape(req.output_shapes[0]);
  o2 = tlo2.template cpu<3>();
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
    ref.data[i] = i1.data[i] * a + i2.data[i];
  }

  Check(o1, ref, EqualEps(1e-6));

  // native and uniform calls should yield bit-exact results
  Check(o1, o2);
}

}  // namespace kernels
}  // namespace dali
