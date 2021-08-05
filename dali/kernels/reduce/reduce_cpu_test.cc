// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <chrono>
#include "dali/kernels/reduce/reduce_cpu.h"

namespace dali {
namespace kernels {

using perfclock = std::chrono::high_resolution_clock;

template <typename Out = double, typename R, typename P>
inline Out microseconds(std::chrono::duration<R, P> d) {
  return std::chrono::duration_cast<std::chrono::duration<Out, std::micro>>(d).count();
}

TEST(ReduceTest, Mean2D) {
  MeanCPU<float, int> mean;

  const int W = 11, H = 7;
  float xmean = (W-1)/2;
  float ymean = 100*(H-1)/2;
  int data[W*H];
  for (int i = 0; i < H; i++)
    for (int j = 0; j < W; j++)
      data[i*W + j] = j + 100*i;
  auto in = make_tensor_cpu<2>(data, { H, W });
  int axes1[] = { 0 };
  int axes2[] = { 1 };
  int axes3[] = { 0, 1 };
  float out[W*H];
  auto out1 = make_tensor_cpu<1>(out, { W });
  mean.Setup(out1, in, make_cspan(axes1));
  mean.Run();
  for (int j = 0; j < W; j++) {
    EXPECT_EQ(out[j], ymean + j);
  }

  auto out2 = make_tensor_cpu<1>(out, { H });
  mean.Setup(out2, in, make_cspan(axes2));
  mean.Run();
  for (int i = 0; i < H; i++) {
    EXPECT_EQ(out[i], i * 100 + xmean);
  }

  auto out3 = make_tensor_cpu<1>(out, { 1 });
  mean.Setup(out3, in, make_cspan(axes3));
  mean.Run();
  EXPECT_EQ(out[0], xmean + ymean);
}

template <typename Reduce, typename Preprocess, typename Postprocess>
void TestStatelessReduction3D(bool mean, Preprocess pre, Postprocess post) {
  Reduce red;

  const int W = 640, H = 480, C = 3;
  std::vector<int> in_v(W*H*C);
  std::vector<float> out_v(W*H*C);
  int *data = in_v.data();
  auto *out_data = out_v.data();

  // fill the array with decreasing values - this will exacerbate numerical instability
  // in sequential approach
  for (int i = 0; i < H; i++)
    for (int j = 0; j < W; j++)
      for (int k = 0; k < C; k++)
        data[(i*W + j) * C + k] = (W-j)*100 + (H-i)*1000 + k;

  SmallVector<int, 3> axes_sets[] = {
    { 0 },
    { 1 },
    { 2 },
    { 0, 1 },
    { 0, 2 },
    { 1, 2 },
    { 0, 1, 2 }
  };

  auto in = make_tensor_cpu<3>(data, { H, W, C });

  std::vector<double> ref(W*H*C);

  for (auto &axes : axes_sets) {
    TensorShape<> out_shape;
    unsigned reduction_mask = 0;
    for (auto a : axes)
      reduction_mask |= (1 << a);
    int64_t den = 1;
    for (int d = 0; d < 3; d++) {
      if (!(reduction_mask & (1u << d))) {
        out_shape.shape.push_back(in.shape[d]);
      } else {
        if (mean)
          den *= in.shape[d];
      }
    }
    if (out_shape.empty())
      out_shape = { 1 };

    auto out = make_tensor_cpu(out_data, out_shape);
    red.Setup(out, in, make_cspan(axes));
    red.Run();

    std::fill(ref.begin(), ref.end(), 0);

    for (int i = 0; i < H; i++) {
      ptrdiff_t ofs_i = reduction_mask&1 ? 0 : i;
      for (int j = 0; j < W; j++) {
        ptrdiff_t ofs_j = reduction_mask&2 ? ofs_i : W*ofs_i+j;
        for (int k = 0; k < C; k++) {
          ptrdiff_t ofs_k = reduction_mask&4 ? ofs_j : C*ofs_j+k;
          auto v = data[(i*W+j)*C+k];
          ref[ofs_k] += pre(v);
        }
      }
    }

    for (int64_t i = 0; i < out.num_elements(); i++) {
      float eps = (ref[i] / den) * 1e-6;
      EXPECT_NEAR(out_v[i], post(ref[i] / den), eps);
    }
  }
}

TEST(ReduceTest, Sum3D) {
  TestStatelessReduction3D<SumCPU<float, int>>(
    false,
    dali::identity(),
    dali::identity());
}

TEST(ReduceTest, Mean3D) {
  TestStatelessReduction3D<MeanCPU<float, int>>(
    true,
    dali::identity(),
    dali::identity());
}

TEST(ReduceTest, MeanSquare3D) {
  TestStatelessReduction3D<MeanSquareCPU<float, int>>(
    true,
    reductions::square(),
    dali::identity());
}

TEST(ReduceTest, RootMeanSquare3D) {
  auto sqrt = [](auto x) { return std::sqrt(x); };
  TestStatelessReduction3D<RootMeanSquareCPU<float, int>>(
    true,
    reductions::square(),
    sqrt);
}

TEST(ReduceTest, StdDev) {
  MeanCPU<float, float> mean;
  StdDevCPU<float, float> stddev;

  std::mt19937_64 rng(1337);
  std::normal_distribution<float> dist(10, 42);

  const int W = 1920, H = 1080;
  std::vector<float> in_v(W*H);
  for (int i = 0; i < H; i++)
    for (int j = 0; j < W; j++)
      in_v[i * W +j] = dist(rng) + i;

  float m1[H] = { 0 };
  float m2[W] = { 0 };
  float m3[1] = { 0 };

  float s1[H] = { 0 };
  float s2[W] = { 0 };
  float s3[1] = { 0 };

  auto in = make_tensor_cpu<2>(in_v.data(), { H, W });
  int axes1[] = { 1 };
  int axes2[] = { 0 };
  int axes3[] = { 0, 1 };

  auto t0 = perfclock::now();
  auto mean1 = make_tensor_cpu<1>(m1, { H });
  mean.Setup(mean1, in, make_span(axes1));
  mean.Run();
  auto t1 = perfclock::now();
  auto stddev1 = make_tensor_cpu<1>(s1, { H });
  stddev.Setup(stddev1, in, make_span(axes1), mean1);
  stddev.Run();
  auto t2 = perfclock::now();
  print(std::cerr, "Performance:\nMean<float, float>: ", microseconds(t1-t0),
                   "us\nStdDev<float, float>: ", microseconds(t2-t1), "us\n");

  for (int i = 0; i < H; i++) {
    EXPECT_NEAR(m1[i], i + 10, 3.5);
    EXPECT_NEAR(s1[i], 42, 3);
  }

  t0 = perfclock::now();
  auto mean2 = make_tensor_cpu<1>(m2, { W });
  mean.Setup(mean2, in, make_span(axes2));
  mean.Run();
  t1 = perfclock::now();
  auto stddev2 = make_tensor_cpu<1>(s2, { W });
  stddev.Setup(stddev2, in, make_span(axes2), mean2);
  stddev.Run();
  t2 = perfclock::now();
  print(std::cerr, "Performance:\nMean<float, float>: ", microseconds(t1-t0),
                   "us\nStdDev<float, float>: ", microseconds(t2-t1), "us\n");

  double dev2 = sqrt(42*42 + H*H/12);  // combined distribution for linear change from 0 to H-1
                                       // and a Gaussian distribution
  for (int i = 0; i < W; i++) {
    EXPECT_NEAR(m2[i], 10 + (H-1)*0.5f, 5);
    EXPECT_NEAR(s2[i], dev2, 5);
  }

  t0 = perfclock::now();
  auto mean3 = make_tensor_cpu<1>(m3, { 1 });
  mean.Setup(mean3, in, make_span(axes3));
  mean.Run();
  t1 = perfclock::now();
  auto stddev3 = make_tensor_cpu<1>(s3, { 1 });
  stddev.Setup(stddev3, in, make_span(axes3), mean3);
  stddev.Run();
  t2 = perfclock::now();
  print(std::cerr, "Performance:\nMean<float, float>: ", microseconds(t1-t0),
                   "us\nStdDev<float, float>: ", microseconds(t2-t1), "us\n");

  EXPECT_NEAR(m3[0], 10 + (H-1)*0.5f, 1);
  EXPECT_NEAR(s3[0], dev2, 1);
}

}  // namespace kernels
}  // namespace dali
