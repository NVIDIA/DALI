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

template <typename T, int ndim>
std::ostream &operator<<(std::ostream &os, const TensorView<StorageCPU, T, ndim> &t) {
  if (!t.data) {
    return os << "[data is null, shape = " << t.shape << "]";
  }
  if (t.dim() == 0) {
    return os << *t.data;
  }

  const char *sep = t.num_elements() > 16 ? ",\n " : ", ";
  os << "[";
  if (t.dim() == 1) {
    for (int64_t i = 0; i < t.num_elements(); i++) {
      if (i)
        os << sep;
      os << t.data[i];
    }
  } else {
    for (int64_t i = 0; i < t.shape[0]; i++) {
      if (i)
        os << sep;
      os << subtensor(t, i);
    }
  }

  return os << "]";
}

template <typename T, int ndim>
std::ostream &operator<<(std::ostream &os, const TensorListView<StorageCPU, T, ndim> &tl) {
  if (tl.data.empty() || !tl.data[0]) {
    return os << "{ data is null, shape = " << tl.shape << " }";
  }
  os << "{ ";
  const char *sep = tl.num_elements() > 16 ? ",\n " : ", ";
  for (int i = 0; i < tl.num_samples(); i++) {
    if (i)
      os << sep;
    os << tl[i];
  }
  return os << " }";
}



TEST(MeanImplGPU, SplitStage) {
  TensorListShape<> in_shape = {{
    { 32, 2, 64000 },
    { 15, 4, 128000 },
    { 72000, 1, 7 }
  }};
  int axes[] = { 0, 2 };
  MeanImplGPU<float, uint8_t> mean;
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
  TestTensorList<float> ref;
  ref.reshape(ref_out_shape);
  auto ref_cpu = ref.cpu();
  for (int i = 0; i < ref_cpu.num_samples(); i++) {
    int64_t n = ref_cpu[i].num_elements();
    int64_t n_in = in_cpu[i].num_elements();
    int64_t ratio = n_in / n;
    RefReduce(ref_cpu[i], in_cpu[i], make_span(axes), reductions::sum());
    for (int j = 0; j < n; j++)
      ref_cpu.data[i][j] /= ratio;
  }

  Check(out_cpu, ref_cpu, EqualEpsRel(1e-5, 1e-6));
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


TEST(StdDevImplGPU, SplitStage) {
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

  TestTensorList<float> centered_square;
  centered_square.reshape(in_shape);
  auto centered_square_cpu = centered_square.cpu();
  for (int i = 0; i < N; i++) {
    auto in_tv = in_cpu[i];
    auto out_tv = centered_square_cpu[i];
    auto mean_tv = mean_cpu[i];
    for (int z = 0; z < in_tv.shape[0]; z++) {
      for (int y = 0; y < in_tv.shape[1]; y++) {
        for (int x = 0; x < in_tv.shape[2]; x++) {
          double d = *in_tv(z, y, x) - *mean_tv(0, y, 0);
          *out_tv(z, y, x) = d * d;
        }
      }
    }
  }

  auto out_cpu = out.cpu(ctx.gpu.stream);
  TestTensorList<float> ref;
  ref.reshape(ref_out_shape);
  auto ref_cpu = ref.cpu();
  for (int i = 0; i < N; i++) {
    int64_t n = ref_cpu[i].num_elements();
    int64_t n_in = in_cpu[i].num_elements();
    double ratio = n_in / n;
    RefReduce(ref_cpu[i], centered_square_cpu[i], make_span(axes), reductions::sum());
    for (int j = 0; j < n; j++)
      ref_cpu.data[i][j] = std::sqrt(ref_cpu.data[i][j] / ratio);
  }

  Check(out_cpu, ref_cpu, EqualEpsRel(1e-5, 1e-6));
}




}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali
