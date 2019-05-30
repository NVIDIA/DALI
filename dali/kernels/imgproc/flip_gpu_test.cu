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

#include "dali/kernels/imgproc/flip_gpu.cuh"
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "dali/kernels/imgproc/flip_test.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/kernels/test/test_tensors.h"
#include "dali/kernels/test/tensor_test_utils.h"

namespace dali {
namespace kernels {

class FlipGpuTest: public testing::TestWithParam<std::array<Index, 4>> {
 public:
  FlipGpuTest()
  : tensor_shape(GetParam())
  , _volume(volume(tensor_shape))
  , shape({tensor_shape, tensor_shape, tensor_shape, tensor_shape,
           tensor_shape, tensor_shape, tensor_shape, tensor_shape}) {}

  void SetUp() override {
    ttl_in.reshape(shape);
    auto tlv = ttl_in.cpu(nullptr);
    std::mt19937_64 rng;
    UniformRandomFill(tlv, rng, 0., 10.);
  }

 protected:
  std::vector<int> flip_x{0, 0, 1, 1, 0, 0, 1, 1};
  std::vector<int> flip_y{0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> flip_z{0, 0, 0, 0, 1, 1, 1, 1};
  TensorShape<4> tensor_shape;
  size_t _volume;
  TensorListShape<4> shape;
  TestTensorList<float, 4> ttl_in;
  TestTensorList<float, 4> ttl_out;
};

TEST_P(FlipGpuTest, BasicTest) {
  GetParam();
  KernelContext ctx;
  FlipGPU<float> kernel;
  auto in_view = ttl_in.gpu(nullptr);
  ttl_in.invalidate_cpu();
  KernelRequirements reqs = kernel.Setup(ctx, in_view);
  ttl_out.reshape(reqs.output_shapes[0].to_static<4>());
  auto out_view = ttl_out.gpu();
  kernel.Run(ctx, out_view, in_view, flip_z, flip_y, flip_x);
  auto out_view_cpu = ttl_out.cpu(nullptr);
  auto in_view_cpu = ttl_in.cpu(nullptr);
  for (int i = 0; i < out_view_cpu.num_samples(); ++i) {
    ASSERT_TRUE(is_flipped(out_view_cpu.tensor_data(i),
              in_view_cpu.tensor_data(i),
              shape[i][0], shape[i][1], shape[i][2], shape[i][3],
              flip_z[i], flip_y[i], flip_x[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(FlipGpuTest, FlipGpuTest,
    ::testing::ValuesIn({
        std::array<Index, 4>{1, 2, 2, 10},
        std::array<Index, 4>{1, 2, 2, 2},
        std::array<Index, 4>{4, 9, 18, 3},
        std::array<Index, 4>{3, 18, 9, 4}}));

}  // namespace kernels
}  // namespace dali
