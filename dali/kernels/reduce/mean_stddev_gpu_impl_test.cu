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
#include <random>
#include <utility>
#include "dali/kernels/reduce/mean_stddev_gpu_impl.cuh"
#include "dali/kernels/scratch.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/reduce/reduce_test.h"

namespace dali {
namespace kernels {
namespace reduce_impl {

TEST(MeanImplGPU, SplitStage) {
  TensorListShape<> in_shape = {{
    { 32, 2, 64000 },
    { 15, 4, 128000 },
    { 72000, 1, 7 }
  }};
  int axes[] = { 0, 2 };
  MeanImplGPU<float, uint8_t, uint64_t> mean;
  KernelContext ctx = {};
  auto req = mean.Setup(ctx, in_shape, make_span(axes), true, false);
  TensorListShape<> ref_out_shape = {{
    { 1, 2, 1 },
    { 1, 4, 1 },
    { 1, 1, 1 }
  }};
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);
  EXPECT_GE(mean.GetNumStages(), 4);  // both reduced axes must be split due to large max extent

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<uint8_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 255);

  TestTensorList<float> out;
  out.reshape(req.output_shapes[0]);
  mean.Run(ctx, out.gpu(), in.gpu());

  auto out_cpu = out.cpu(ctx.gpu.stream);
  TestTensorList<int64_t> ref_sum;
  TestTensorList<float> ref;
  ref.reshape(ref_out_shape);
  ref_sum.reshape(ref_out_shape);
  auto ref_cpu = ref.cpu();
  auto ref_sum_cpu = ref.cpu();
  for (int i = 0; i < ref_cpu.num_samples(); i++) {
    int64_t n = ref_cpu[i].num_elements();
    int64_t n_in = in_cpu[i].num_elements();
    int64_t ratio = n_in / n;
    RefReduce(ref_sum_cpu[i], in_cpu[i], make_span(axes), reductions::sum());
    for (int j = 0; j < n; j++)
      ref_cpu.data[i][j] = ref_sum_cpu.data[i][j] / ratio;
  }

  Check(out_cpu, ref_cpu, EqualEpsRel(1e-6, 1e-6));
}



TEST(MeanImplGPU, BatchMean) {
  TensorListShape<> in_shape = {{
    { 32, 3, 64000 },
    { 15, 3, 128000 },
    { 72000, 3, 7 }
  }};
  int axes[] = { 0, 2 };
  MeanImplGPU<float, uint8_t> mean;
  KernelContext ctx = {};
  auto req = mean.Setup(ctx, in_shape, make_span(axes), false, true);
  TensorListShape<> ref_out_shape = {{
    TensorShape<>{3}
  }};
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);
  EXPECT_GE(mean.GetNumStages(), 4);  // both reduced axes must be split due to large max extent

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<uint8_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 255);

  TestTensorList<float> out;
  out.reshape(req.output_shapes[0]);
  mean.Run(ctx, out.gpu(), in.gpu());

  auto out_cpu = out.cpu(ctx.gpu.stream);
  TestTensorList<float> ref;
  TestTensorList<int64_t> ref_samples;
  TensorListShape<> ref_samples_shape = {{
    { 1, 3, 1 },
    { 1, 3, 1 },
    { 1, 3, 1 }
  }};
  ref.reshape(ref_out_shape);
  ref_samples.reshape(ref_samples_shape);
  auto ref_cpu = ref.cpu();
  auto ref_samples_cpu = ref_samples.cpu();
  int N = ref_samples_shape.num_samples();
  for (int i = 0; i < N; i++) {
    RefReduce(ref_samples_cpu[i], in_cpu[i], make_span(axes), reductions::sum());
  }
  int64_t n = ref_out_shape.num_elements();
  double ratio = in_cpu.num_elements() / n;
  for (int j = 0; j < n; j++) {
    int64_t sum = 0;
    for (int i = 0; i < N; i++)
      sum += ref_samples_cpu.data[i][j];
    ref_cpu.data[0][j] = sum / ratio;
  }

  Check(out_cpu, ref_cpu, EqualEpsRel(1e-5, 1e-6));
}

template <typename Out, typename In, typename Mean>
void CenterAndSquare(const OutTensorCPU<Out> &out,
                     const InTensorCPU<In> &in,
                     const InTensorCPU<Mean> mean,
                     TensorShape<> &in_pos,
                     TensorShape<> &mean_pos,
                     int dim = 0) {
  int extent = in.shape[dim];
  int dj = mean.shape[dim] > 1 ? 1 : 0;
  if (dim == in.dim() - 1) {
    const Mean *mean_ptr = mean(mean_pos);
    const In *in_ptr = in(in_pos);
    Out *out_ptr = out(in_pos);
    for (int i = 0, j = 0; i < extent; i++, j += dj) {
      double d = in_ptr[i] - mean_ptr[j];
      out_ptr[i] = static_cast<Out>(d * d);
    }
  } else {
    for (int i = 0, j = 0; i < extent; i++, j += dj) {
      in_pos[dim] = i;
      mean_pos[dim] = j;
      CenterAndSquare(out, in, mean, in_pos, mean_pos, dim + 1);
    }
  }
}

template <typename Out = float, typename In, typename Mean>
TestTensorList<Out> CenterAndSquare(const InListCPU<In> &in,
                                    const InListCPU<Mean> &mean) {
  TestTensorList<Out> out_tl;
  out_tl.reshape(in.shape);
  auto out = out_tl.cpu();
  int N = in.num_samples();
  for (int i = 0; i < N; i++) {
    auto in_tv = in[i];
    auto out_tv = out[i];
    auto mean_tv = mean.num_samples() > 1 ? mean[i] : mean[0];
    TensorShape<> in_pos, mean_pos;
    in_pos.resize(in_tv.shape.size());
    mean_pos.resize(in_tv.shape.size());
    CenterAndSquare(out_tv, in_tv, mean_tv, in_pos, mean_pos);
  }
  return out_tl;
}

template <typename Out = float, typename In, typename Mean>
TestTensorList<Out> RefStdDev(const TensorListView<StorageCPU, In> &in,
                              const TensorListView<StorageCPU, Mean> &mean,
                              double reg = 0, bool inv = false) {
  reg *= reg;
  SmallVector<int, 6> axes;
  for (int d = 0; d < mean.sample_dim(); d++) {
    for (int i = 0; i < mean.num_samples(); i++) {
      if (mean.tensor_shape_span(i)[d] > 1)
        goto non_reduced;
    }
    axes.push_back(d);
  non_reduced:;  // NOLINT
  }

  bool reduce_batch = mean.num_samples() == 1 && in.num_samples() > 1;

  using tmp_t = decltype(In() - Mean());
  auto centered_squared = CenterAndSquare<tmp_t, In, Mean>(in, mean);
  auto centered_squared_cpu = centered_squared.cpu();
  TestTensorList<Out> reduced_samples;

  const auto &out_shape = mean.shape;

  int N = in.num_samples();

  if (reduce_batch) {
    assert(is_uniform(out_shape));
    TensorListShape<> reduced_sample_shapes = uniform_list_shape(in.num_samples(), mean.shape[0]);
    reduced_samples.reshape(reduced_sample_shapes);
    auto reduced_samples_cpu = reduced_samples.cpu();

    for (int i = 0; i < N; i++) {
      RefReduce(reduced_samples_cpu[i], centered_squared_cpu[i],
                make_span(axes), reductions::sum());
    }

    TestTensorList<Out> out_tl;
    out_tl.reshape(out_shape);
    auto out = out_tl.cpu();
    int64_t n = out_shape.num_elements();
    double ratio = in.num_elements() / n;
    for (int j = 0; j < n; j++) {
      double sum = reg;
      for (int i = 0; i < N; i++)
        sum += reduced_samples_cpu.data[i][j];
      out.data[0][j] = inv ? rsqrt(sum / ratio) : std::sqrt(sum / ratio);
    }
    return out_tl;
  } else {
    reduced_samples.reshape(out_shape);
    auto out = reduced_samples.cpu();
    for (int i = 0; i < N; i++) {
      int64_t n = out[i].num_elements();
      int64_t n_in = in[i].num_elements();
      double ratio = n_in / n;
      RefReduce(out[i], centered_squared_cpu[i], make_span(axes), reductions::sum());
      for (int j = 0; j < n; j++) {
        double x = (out.data[i][j] + reg) / ratio;
        out.data[i][j] = inv ? rsqrt(x) : std::sqrt(x);
      }
    }
    return reduced_samples;
  }
}


TEST(StdDevImplGPU, Outer_Inner_SplitStage) {
  TensorListShape<> in_shape = {{
    { 32, 2, 64000 },
    { 15, 4, 128000 },
    { 72000, 1, 7 }
  }};
  int axes[] = { 0, 2 };
  StdDevImplGPU<float, int16_t> stddev;
  KernelContext ctx = {};
  auto req = stddev.Setup(ctx, in_shape, make_span(axes), true, false);
  TensorListShape<> ref_out_shape = {{
    { 1, 2, 1 },
    { 1, 4, 1 },
    { 1, 1, 1 }
  }};
  int N = in_shape.num_samples();
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);
  EXPECT_GE(stddev.GetNumStages(), 4);  // both reduced axes must be split due to large max extent

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<int16_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, -100, 100);

  TestTensorList<float> fake_mean;
  fake_mean.reshape(ref_out_shape);
  auto mean_cpu = fake_mean.cpu();
  *mean_cpu[0](0, 0, 0) = 10;
  *mean_cpu[0](0, 1, 0) = 20;
  *mean_cpu[1](0, 0, 0) = 30;
  *mean_cpu[1](0, 1, 0) = 40;
  *mean_cpu[1](0, 2, 0) = 50;
  *mean_cpu[1](0, 3, 0) = 60;
  *mean_cpu[2](0, 0, 0) = 70;


  TestTensorList<float> out;
  out.reshape(req.output_shapes[0]);
  stddev.Run(ctx, out.gpu(), in.gpu(), fake_mean.gpu());
  auto out_cpu = out.cpu(ctx.gpu.stream);

  TestTensorList<float> ref = RefStdDev(in_cpu, mean_cpu);

  Check(out.cpu(), ref.cpu(), EqualEpsRel(1e-5, 1e-6));
}


TEST(StdDevImplGPU, Middle_Inner_Sample) {
  TensorListShape<> in_shape = {{
    { 4, 32, 1, 6400 },
    { 3, 15, 2, 12800 },
    { 2, 7200, 3, 7 }
  }};
  int axes[] = { 1, 3 };
  StdDevImplGPU<float, int16_t> stddev;
  KernelContext ctx = {};
  auto req = stddev.Setup(ctx, in_shape, make_span(axes), true, false);
  TensorListShape<> ref_out_shape = {{
    { 4, 1, 1, 1 },
    { 3, 1, 2, 1 },
    { 2, 1, 3, 1 }
  }};

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<int16_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, -100, 100);

  TestTensorList<float> fake_mean;
  fake_mean.reshape(ref_out_shape);
  auto mean_cpu = fake_mean.cpu();
  for (int i = 0, n = mean_cpu.num_elements(); i < n; i++) {
    mean_cpu.data[0][i] = 10 * (i+1);
  }

  TestTensorList<float> out;
  out.reshape(req.output_shapes[0]);
  stddev.Run(ctx, out.gpu(), in.gpu(), fake_mean.gpu());
  auto out_cpu = out.cpu(ctx.gpu.stream);

  TestTensorList<float> ref = RefStdDev(in_cpu, mean_cpu);

  Check(out_cpu, ref.cpu(), EqualEpsRel(1e-5, 1e-6));
}

TEST(StdDevImplGPU, Middle_Inner_Batch) {
  TensorListShape<> in_shape = {{
    { 2, 32, 3, 6400 },
    { 2, 15, 3, 12800 },
    { 2, 7200, 3, 7 }
  }};
  int axes[] = { 1, 3 };
  StdDevImplGPU<float, int16_t> stddev;
  KernelContext ctx = {};
  auto req = stddev.Setup(ctx, in_shape, make_span(axes), true, true);
  TensorListShape<> ref_out_shape = {{
    { 2, 1, 3, 1 }
  }};

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<int16_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, -100, 100);

  TestTensorList<float> fake_mean;
  fake_mean.reshape(ref_out_shape);
  auto mean_cpu = fake_mean.cpu();
  *mean_cpu[0](0, 0, 0, 0) = 10;
  *mean_cpu[0](0, 0, 1, 0) = 20;
  *mean_cpu[0](0, 0, 2, 0) = 30;
  *mean_cpu[0](1, 0, 0, 0) = 40;
  *mean_cpu[0](1, 0, 1, 0) = 50;
  *mean_cpu[0](1, 0, 2, 0) = 60;

  TestTensorList<float> out;
  out.reshape(req.output_shapes[0]);
  stddev.Run(ctx, out.gpu(), in.gpu(), fake_mean.gpu());
  auto out_cpu = out.cpu(ctx.gpu.stream);

  TestTensorList<float> ref = RefStdDev(in_cpu, mean_cpu);

  Check(out_cpu, ref.cpu(), EqualEpsRel(1e-5, 1e-6));
}


TEST(InvStdDevImplGPU, Outer_Batch_Regularized) {
  TensorListShape<> in_shape = {{
    { 480, 640, 3 },
    { 720, 1280, 3 },
    { 1080, 1920, 3 }
  }};
  int axes[] = { 0, 1 };
  InvStdDevImplGPU<float, int16_t> stddev;
  KernelContext ctx = {};
  auto req = stddev.Setup(ctx, in_shape, make_span(axes), true, true);
  TensorListShape<> ref_out_shape = {{
    { 1, 1, 3 }
  }};

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<int16_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, -100, 100);

  TestTensorList<float> fake_mean;
  fake_mean.reshape(ref_out_shape);
  auto mean_cpu = fake_mean.cpu();
  for (int i = 0, n = mean_cpu.num_elements(); i < n; i++) {
    mean_cpu.data[0][i] = 10 * (i+1);
  }

  TestTensorList<float> out;
  out.reshape(req.output_shapes[0]);
  stddev.Run(ctx, out.gpu(), in.gpu(), fake_mean.gpu(), 100000);
  auto out_cpu = out.cpu(ctx.gpu.stream);

  TestTensorList<float> ref = RefStdDev(in_cpu, mean_cpu, 100000, true);

  Check(out_cpu, ref.cpu(), EqualEpsRel(1e-5, 1e-6));
}


}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali
