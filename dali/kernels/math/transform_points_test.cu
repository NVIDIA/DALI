// Copyright (c) 2020, 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <vector>
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/math/transform_points.h"
#include "dali/kernels/math/transform_points.cuh"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {

struct TransformPointsTest : ::testing::Test {
  using In = uint8_t;
  using Out = uint16_t;

  static const int in_dim = 3;
  static const int out_dim = 2;

  void PrepareData() {
    TensorListShape<3> shape = {{
      { 480, 640, in_dim },
      { 100, 120, in_dim }
    }};

    TensorListShape<3> out_shape = {{
      { 480, 640, out_dim },
      { 100, 120, out_dim }
    }};

    in_data_.reshape(shape);
    out_data_.reshape(out_shape);

    UniformRandomFill(in_data_.cpu(), rng_, 0, 255);
  }

  void RunCPU() {
    PrepareData();
    using Kernel = TransformPointsCPU<Out, In, out_dim, in_dim>;
    mat<out_dim, in_dim> M;
    vec<out_dim> T;
    auto dist = uniform_distribution<float>(-0.5, 0.5);
    auto t_dist = uniform_distribution<float>(min_value<In>() / 2, max_value<In>() / 2);
    for (int i = 0; i < out_dim; i++) {
      for (int j = 0; j < in_dim; j++) {
        M(i, j) = dist(rng_) + (i == j);  // random + diagonal 1
      }
      T[i] = t_dist(rng_);
    }
    auto in = in_data_.cpu();
    auto out = out_data_.cpu();
    kmgr_.Resize<Kernel>(1);
    double eps = std::is_integral<Out>::value ? 0.501 : 1e-3;
    for (int i = 0; i < in.num_samples(); i++) {
      TensorView<StorageCPU, const In> in_tensor = in[i];
      TensorView<StorageCPU, Out> out_tensor = out[i];
      const auto *in_points = reinterpret_cast<const vec<in_dim, In> *>(in_tensor.data);
      const auto *out_points = reinterpret_cast<const vec<out_dim, Out> *>(out_tensor.data);
      KernelContext ctx;
      ctx.gpu.stream = 0;
      auto &req = kmgr_.Setup<Kernel>(0, ctx, in_tensor.shape);
      ASSERT_EQ(req.output_shapes[0][0], out_tensor.shape);
      kmgr_.Run<Kernel>(0, ctx, out_tensor, in_tensor, M, T);
      for (int j = 0; j < in_tensor.shape[0]; j++) {
        vec<out_dim> ref = M * in_points[j] + T;
        for (int d = 0; d < out_dim; d++) {
          float r = clamp<float>(ref[d], min_value<Out>(), max_value<Out>());
          EXPECT_NEAR(r, out_points[j][d], eps);
        }
      }
    }
  }

  void RunGPU() {
    PrepareData();
    using Kernel = TransformPointsGPU<Out, In, out_dim, in_dim>;
    auto dist = uniform_distribution<float>(-0.5, 0.5);
    auto t_dist = uniform_distribution<float>(min_value<In>() / 2, max_value<In>() / 2);
    auto in_gpu = in_data_.gpu();
    auto out_gpu = out_data_.gpu();
    int N = in_gpu.num_samples();
    std::vector<mat<out_dim, in_dim>> M(N);
    std::vector<vec<out_dim>> T(N);
    for (int s = 0; s < N; s++)
      for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
          M[s](i, j) = dist(rng_) + (i == j);  // random + diagonal 1
        }
        T[s][i] = t_dist(rng_);
      }

    kmgr_.Resize<Kernel>(1);
    KernelContext ctx;
    ctx.gpu.stream = 0;
    auto &req = kmgr_.Setup<Kernel>(0, ctx, in_gpu.shape);
    ASSERT_EQ(req.output_shapes[0], out_gpu.shape);
    kmgr_.Run<Kernel>(0, ctx, out_gpu, in_gpu, make_span(M), make_span(T));

    auto in_cpu = in_data_.cpu();
    auto out_cpu = out_data_.cpu();

    double eps = std::is_integral<Out>::value ? 0.501 : 1e-3;
    for (int i = 0; i < N; i++) {
      TensorView<StorageCPU, const In> in_tensor = in_cpu[i];
      TensorView<StorageCPU, Out> out_tensor = out_cpu[i];
      const auto *in_points = reinterpret_cast<const vec<in_dim, In> *>(in_tensor.data);
      const auto *out_points = reinterpret_cast<const vec<out_dim, Out> *>(out_tensor.data);
      for (int j = 0; j < in_tensor.shape[0]; j++) {
        vec<out_dim> ref = M[i] * in_points[j] + T[i];
        for (int d = 0; d < out_dim; d++) {
          float r = clamp<float>(ref[d], min_value<Out>(), max_value<Out>());
          EXPECT_NEAR(r, out_points[j][d], eps);
        }
      }
    }
  }

  TestTensorList<In> in_data_;
  TestTensorList<Out> out_data_;
  KernelManager kmgr_;

  std::mt19937_64 rng_{1234};
};

TEST_F(TransformPointsTest, CPU) {
  RunCPU();
}

TEST_F(TransformPointsTest, GPU) {
  RunGPU();
}

}  // namespace kernels
}  // namespace dali
