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

#ifndef DALI_KERNELS_TEST_KERNEL_POC_TEST_H_
#define DALI_KERNELS_TEST_KERNEL_POC_TEST_H_

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "dali/kernels/kernel.h"
#include "dali/test/test_tensors.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/test/kernel_test_utils.h"

namespace dali {
namespace kernels {

template <typename StorageBackend, typename Kernel,
          typename Base = dali::testing::SimpleKernelTestBase<Kernel>>
struct KernelPoCFixture : Base {
 public:
  using Input1 = typename Base::template InputElement<0>;
  using Input2 = typename Base::template InputElement<1>;
  using Output = typename Base::template OutputElement<0>;
  void RunImpl() {
    ASSERT_NO_FATAL_FAILURE(Initialize());
    ASSERT_NO_FATAL_FAILURE(CalcReference());
    ASSERT_NO_FATAL_FAILURE(Launch());
    ASSERT_NO_FATAL_FAILURE(Verify());
  }

 private:
  KernelContext ctx;
  Kernel kernel;
  TestTensorList<Input1> tl1;
  TestTensorList<Input2> tl2;
  TestTensorList<Output> tlo1, tlo2, tlref;
  TensorListShape<3> list_shape;
  float a = 0.5f;

  InList<StorageBackend, Input1, 3> i1;
  InList<StorageBackend, Input2, 3> i2;
  OutList<StorageBackend, Output, 3> o1, o2;


  void Initialize() {
    std::mt19937_64 rng;

    std::vector<TensorShape<3>> shapes;
    auto size_dist = uniform_distribution(128, 256);
    for (int i = 0; i < 3; i++)
      shapes.push_back({ i+1, size_dist(rng), size_dist(rng)});
    list_shape = shapes;

    tl1.reshape(list_shape);
    tl2.reshape(list_shape);

    Input1 max1 = std::is_integral<Input1>::value ? 100 : 1;
    Input2 max2 = std::is_integral<Input2>::value ? 100 : 1;
    UniformRandomFill(tl1.template cpu<3>(0), rng, 0, max1);
    UniformRandomFill(tl2.template cpu<3>(0), rng, 0, max2);
  }

  void CalcReference() {
    auto i1_cpu = tl1.template cpu<3>();
    auto i2_cpu = tl2.template cpu<3>();

    tlref.reshape(list_shape);
    auto ref = tlref.template cpu<3>(0);

    for (int sample = 0; sample < ref.num_samples(); sample++) {
      ptrdiff_t elements = ref.tensor_shape(sample).num_elements();
      for (ptrdiff_t i = 0; i < elements; i++) {
        ref.tensor_data(sample)[i] =
          i1_cpu.tensor_data(sample)[i] * a + i2_cpu.tensor_data(sample)[i];
      }
    }
  }

  void Launch() {
    i1 = tl1.template get<StorageBackend, 3>();
    i2 = tl2.template get<StorageBackend, 3>();

    auto req = kernel.Setup(ctx, i1, i2, a);
    ASSERT_EQ((int)req.output_shapes.size(), 1);
    ASSERT_NO_FATAL_FAILURE(CheckEqual(req.output_shapes[0], i1.shape));

    // Kernel's native Run
    tlo1.reshape(req.output_shapes[0]);
    o1 = tlo1.template get<StorageBackend, 3>();
    kernel.Run(ctx, o1, i1, i2, a);

    // use uniform call with argument tuples
    tlo2.reshape(req.output_shapes[0]);
    o2 = tlo2.template get<StorageBackend, 3>();
    kernels::kernel::Run(kernel, ctx, std::tie(o2), std::tie(i1, i2), std::make_tuple(a) );
  }

  void Verify() {
    // verify that shape hasn't changed
    ASSERT_NO_FATAL_FAILURE(CheckEqual(o1.shape, i1.shape));
    ASSERT_NO_FATAL_FAILURE(CheckEqual(o2.shape, i1.shape));

    auto o1_cpu = tlo1.template cpu<3>();
    auto o2_cpu = tlo2.template cpu<3>();

    auto ref = tlref.template cpu<3>(0);
    Check(o1_cpu, ref, EqualEps(1e-6));

    // native and uniform calls should yield bit-exact results
    Check(o1_cpu, o2_cpu);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TEST_KERNEL_POC_TEST_H_
