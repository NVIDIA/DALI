// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <string>
#include <tuple>
#include "dali/kernels/normalize/normalize_cpu.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {

template <typename Out, typename In, typename Params, int ndim>
void ReferenceNormalize(
    const OutTensorCPU<Out, ndim> &out,
    const InTensorCPU<In, ndim> &in,
    const InTensorCPU<Params, ndim> &mean,
    const InTensorCPU<Params, ndim> &inv_stddev,
    TensorShape<ndim> &data_pos,
    TensorShape<ndim> &param_pos,
    int axis) {
  int param_step = mean.shape[axis] == 1 ? 0 : 1;
  if (axis == in.dim() - 1) {
    for (int i = 0, j = 0; i < in.shape[axis]; i++, j += param_step) {
      data_pos[axis] = i;
      param_pos[axis] = j;
      auto m = *mean(param_pos);
      auto d = *inv_stddev(param_pos);
      auto x = *in(data_pos);
      *out(data_pos) = ConvertSat<Out>((x - m) * d);
    }
  } else {
    for (int i = 0, j = 0; i < in.shape[axis]; i++, j += param_step) {
      data_pos[axis] = i;
      param_pos[axis] = j;
      ReferenceNormalize(out, in, mean, inv_stddev, data_pos, param_pos, axis+1);
    }
  }
}

template <typename Out, typename In, typename Params, int ndim>
void ReferenceNormalize(
    const OutTensorCPU<Out, ndim> &out,
    const InTensorCPU<In, ndim> &in,
    const InTensorCPU<Params, ndim> &mean,
    const InTensorCPU<Params, ndim> &inv_stddev) {
  assert(out.shape == in.shape);
  assert(mean.shape == inv_stddev.shape);
  TensorShape<ndim> data_pos, param_pos;
  data_pos.resize(in.dim());
  param_pos.resize(mean.dim());
  ReferenceNormalize(out, in, mean, inv_stddev, data_pos, param_pos, 0);
}

TEST(NormalizeTest, 1D_elementwise) {
  constexpr int N = 8;
  const float data[N] = {
    1, 2, 3, 4, 5, 6, 7, 8
  };
  const float mean[N] = {
    3, 2, 1, 3, 2, 1, 3, 2
  };
  const float inv_stddev[N] = {
    2, 3, 4, 5, 6, 7, 8, 9
  };
  float out_data[N] = {}, ref_data[N];
  NormalizeCPU<float, float> norm;
  auto in = make_tensor_cpu<1>(data, { N });
  auto M = make_tensor_cpu<1>(mean, { N });
  auto D = make_tensor_cpu<1>(inv_stddev, { N });
  auto out = make_tensor_cpu<1>(out_data, in.shape);
  auto ref = make_tensor_cpu<1>(ref_data, out.shape);
  ReferenceNormalize(ref, in, M, D);
  KernelContext ctx;
  norm.Setup(ctx, in.shape, M.shape);
  norm.Run(ctx, out, in, M, D);
  Check(out, ref);
}

TEST(NormalizeTest, 1D_global) {
  constexpr int N = 8;
  const float data[N] = {
    1, 2, 3, 4, 5, 6, 7, 8
  };
  const float mean = 2;
  const float inv_stddev = 3;
  NormalizeCPU<float, float> norm;
  float out_data[N] = {}, ref_data[N] = {};
  auto in = make_tensor_cpu<1>(data, { N });
  auto M = make_tensor_cpu<1>(&mean, { 1 });
  auto D = make_tensor_cpu<1>(&inv_stddev, { 1 });
  auto out = make_tensor_cpu<1>(out_data, in.shape);
  auto ref = make_tensor_cpu<1>(ref_data, out.shape);
  ReferenceNormalize(ref, in, M, D);
  KernelContext ctx;
  norm.Setup(ctx, in.shape, M.shape);
  norm.Run(ctx, out, in, M, D);
  Check(out, ref);
}

class NormalizeNDTest : public ::testing::Test,
                        public ::testing::WithParamInterface<std::tuple<int, int>> {
 public:
  NormalizeNDTest()
  : rng(1337)
  , dim(std::get<0>(GetParam()))
  , mask(std::get<1>(GetParam())) {
    data_shape.resize(dim);
    param_shape.resize(dim);
    mask_str = std::string(dim, '-');
    for (int i = 0; i < dim; i++)
      if (is_reduced_dim(i))
        mask_str[i] = 'X';
  }

 protected:
  void SetupIter() {
    int maxd = 8;
    const int maxd_by_dim[] = { 10000, 1000, 100, 10, 10, 8 };
    if (dim < 6)
      maxd = maxd_by_dim[dim];
    UniformRandomFill(data_shape, rng, 1, maxd);
    for (int i = 0; i < dim; i++) {
      param_shape[i] = is_reduced_dim(i) ? 1 : data_shape[i];
    }
    out_data.clear();
    ref_data.clear();
    in_data.resize(data_shape.num_elements());
    out_data.resize(data_shape.num_elements(), std::nanf(""));
    ref_data.resize(data_shape.num_elements(), std::nanf(""));
    mean_data.resize(param_shape.num_elements());
    invstddev_data.resize(param_shape.num_elements());
    UniformRandomFill(in_data, rng, -10.0, 10.0);
    UniformRandomFill(mean_data, rng, -10.0, 10.0);
    UniformRandomFill(invstddev_data, rng, 0.0, 10.0);
  }

  bool is_reduced_dim(int d) const noexcept { return (mask & (1 << d)) != 0; }

  std::mt19937 rng;
  const int dim, mask;
  TensorShape<> data_shape, param_shape;
  std::string mask_str;
  std::vector<float> in_data, out_data, ref_data, mean_data, invstddev_data;
};

TEST_P(NormalizeNDTest, NormalizeNDTest) {
  const int max_iter = 10;
  for (int i = 0; i < max_iter; i++) {
    SetupIter();
    auto in = make_tensor_cpu(in_data.data(), data_shape);
    auto out = make_tensor_cpu(out_data.data(), data_shape);
    auto ref = make_tensor_cpu(ref_data.data(), data_shape);
    auto mean = make_tensor_cpu(mean_data.data(), param_shape);
    auto invstddev = make_tensor_cpu(invstddev_data.data(), param_shape);
    NormalizeCPU<float, float> norm;
    KernelContext ctx;
    auto req = norm.Setup(ctx, data_shape, param_shape);
    ASSERT_EQ(req.output_shapes[0][0], data_shape);
    norm.Run(ctx, out, in, mean, invstddev);
    ReferenceNormalize<float, float, float, -1>(ref, in, mean, invstddev);
    Check(out, ref);  // for CPU, we expect bit-exact result
    if (HasFailure()) {
      FAIL() << "Test failed with dim = " << dim
             << " reduction mask = " << mask_str
             << "\ndata shape      " << data_shape
             << "\nparameter shape " << param_shape;
    }
  }
}

static auto GetDimsAndMasks(int min_dim, int max_dim) {
  std::vector<std::tuple<int, int>> ret;
  for (int dim = min_dim; dim <= max_dim; dim++) {
    for (int mask = 0; mask < (1 << dim); mask++) {
      ret.emplace_back(dim, mask);
    }
  }
  return ret;
}

static auto dim_and_masks = GetDimsAndMasks(1, 6);

INSTANTIATE_TEST_SUITE_P(NormalizeNDTest, NormalizeNDTest, testing::ValuesIn(dim_and_masks));

}  // namespace kernels
}  // namespace dali
