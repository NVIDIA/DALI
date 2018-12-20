// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <dali/kernels/kernel.h>
#include <dali/kernels/test/test_tensors.h>
#include <dali/kernels/tensor_shape_str.h>
#include <gtest/gtest.h>
#include <dali/kernels/test/tensor_test_utils.h>
#include <random>

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


template <typename Input1, typename Input2, typename Output>
void RunTest() {
  unsigned seed = 123;
  MADKernel<Input1, Input2, Output> K;

  InListCPU<Input1, 3> i1;
  InListCPU<Input2, 3> i2;
  OutListCPU<Output, 3> o;

  KernelContext ctx;

  TestTensorList<Input1> tl1;
  TestTensorList<Input2> tl2;
  TestTensorList<Output> tlo, tlref;

  std::vector<TensorShape<3>> shapes;
  for (int i = 0; i < 3; i++)
    shapes.push_back({ i+1, rand_r(&seed)%512+512, rand_r(&seed)%512+512});
  TensorListShape<3> list_shape(shapes);

  tl1.reshape(list_shape);
  tl2.reshape(list_shape);

  RandomFill(tl1.template cpu<3>(0), 0, 1);
  RandomFill(tl2.template cpu<3>(0), 0, 1);

  i1 = tl1.template cpu<3>();
  i2 = tl2.template cpu<3>();

  float a = 0.5f;

  auto req = K.GetRequirements(ctx, i1, i2, a);
  ASSERT_EQ(req.output_shapes.size(), 1);
  tlo.reshape(req.output_shapes[0]);
  o = tlo.template cpu<3>();
  K.Run(ctx, o, i1, i2, 0.5f);

  ASSERT_NO_FATAL_FAILURE(CheckEqual(o.shape, i1.shape));

  tlref.reshape(list_shape);
  auto ref = tlref.template cpu<3>(0);
  ptrdiff_t total = ref.num_elements();

  for (ptrdiff_t i = 0; i < total; i++) {
    ref.data[i] = i1.data[i] * a + i2.data[i];  // trivial elementwise
  }

  Check(o, ref);
}

TEST(KernelPoC, MAD) {
  RunTest<float, float, float>();
}

}  // namespace kernels
}  // namespace dali
