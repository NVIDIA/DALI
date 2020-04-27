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
#include "dali/kernels/reduce/reduce_gpu_impl.cuh"
#include "dali/kernels/scratch.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/reduce/reduce_test.h"

namespace dali {
namespace kernels {
namespace reduce_impl {

TEST(ReduceImplGPU, SimplifyComplex) {
  TensorListShape<> tls = {{
    { 1, 2, 1, 1, 1, 3, 3 },
    { 1, 1, 3, 1, 5, 2, 3 }
  //  T  F  F  T  F  F, F    <-- all samples with extent == 1
  }};
  EXPECT_TRUE(is_degenerate_dim(tls, 0));
  EXPECT_TRUE(is_degenerate_dim(tls, 3));
  SmallVector<int, 6> out_axes;
  SmallVector<std::pair<int, int>, 6> groups;

  int axes[] = { 0, 2, 4, 6 };
  // axes 0, 2, 4 and 6
  //
  // axis 0 is degenerate, so will not be reduced (no need to) - and will be collapsed
  // axis 3 is not reduced, but is degenerate, so it should be merged with 2 and 4 (both reduced)
  // axis 6 is reduced
  //
  // Axis 0 will disappear
  // Axis 1 -> renamed to 0
  // Axes 2-4 will be merged into one axis, 1
  // Axis 5 -> renamed to 2
  // Axis 6 -> renamed to 3

  SimplifyReduction(out_axes, groups, tls, make_span(axes));
  ASSERT_EQ(out_axes.size(), 2);
  EXPECT_EQ(out_axes[0], 1);
  EXPECT_EQ(out_axes[1], 3);
  EXPECT_EQ(groups.size(), 4);
  EXPECT_EQ(groups[0], std::make_pair(0, 2));  // 2 axes grouped
  EXPECT_EQ(groups[1], std::make_pair(2, 3));  // 3 axes grouped
  EXPECT_EQ(groups[2], std::make_pair(5, 1));  // ungrouped
  EXPECT_EQ(groups[3], std::make_pair(6, 1));  // ungrouped
}

TEST(ReduceImplGPU, Simplify_NoReduce) {
  for (int axis = 0; axis < 3; axis++) {
    TensorListShape<> tls = {{
      { 2, 3, 4 },
      { 3, 4, 5 }
    }};

    for (int i = 0; i < tls.num_samples(); i++)
      tls.tensor_shape_span(i)[axis] = 1;

    for (int a = 0; a < 3; a++)
      EXPECT_EQ(is_degenerate_dim(tls, a), a == axis);

    int axes[] = { axis };
    SmallVector<int, 6> out_axes;
    SmallVector<std::pair<int, int>, 6> groups;
    SimplifyReduction(out_axes, groups, tls, make_span(axes));
    EXPECT_TRUE(out_axes.empty());
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0], std::make_pair(0, 3));
  }
}

TEST(ReduceImplGPU, Simplify_NoOp) {
  TensorListShape<> tls = {{
    { 2, 3, 4 },
    { 3, 4, 5 }
  }};
  int axes[] = { 1 };
  SmallVector<int, 6> out_axes;
  SmallVector<std::pair<int, int>, 6> groups;
  SimplifyReduction(out_axes, groups, tls, make_span(axes));
  ASSERT_EQ(out_axes.size(), 1u);
  EXPECT_EQ(out_axes[0], 1);
  ASSERT_EQ(groups.size(), 3u);
  EXPECT_EQ(groups[0], std::make_pair(0, 1));
  EXPECT_EQ(groups[1], std::make_pair(1, 1));
  EXPECT_EQ(groups[2], std::make_pair(2, 1));
}

TEST(ReduceImpl, TestCheckAxes) {
  EXPECT_NO_THROW(CheckAxes({}, 0));
  int axes_0[] = { 0 };
  int axes_01[] = { 0, 1 };
  int axes_2[] = { 2 };
  int axes_010[] = { 0, 1, 0 };
  EXPECT_NO_THROW(CheckAxes(make_span(axes_0), 1));
  EXPECT_NO_THROW(CheckAxes(make_span(axes_01), 2));
  EXPECT_THROW(CheckAxes(make_span(axes_2), 2), std::out_of_range);
  EXPECT_THROW(CheckAxes(make_span(axes_010), 2), std::invalid_argument);
}

TEST(ReduceImpl, TestCheckBatchReduce) {
  TensorListShape<> tls = {{
    { 3, 3, 2, 4 },
    { 3, 4, 2, 5 },
    { 3, 5, 2, 6 }
  }};

  unsigned must_reduce = 0b1010;  // dimensions 1 and 3 are non-uniform and must be reduced

  SmallVector<int, 4> axes;
  for (unsigned mask = 0; mask < 16; mask++) {
    axes.clear();
    for (int a = 0; a < 4; a++) {
      if (mask & (1 << a))
        axes.push_back(a);
    }
    if ((mask & must_reduce) == must_reduce) {
      EXPECT_NO_THROW(CheckBatchReduce(tls, make_span(axes)));
    } else {
      EXPECT_THROW(CheckBatchReduce(tls, make_span(axes)), std::exception);
    }
  }
}

TEST(SumImplGPU, Inner) {
  TensorListShape<> in_shape = {{
    { 3, 480, 640 },
    { 3, 720, 1280 },
    { 1, 576, 720 }
  }};
  int axes[] = { 1, 2 };
  SumImplGPU<int64_t, uint8_t> sum;
  KernelContext ctx = {};
  auto req = sum.Setup(ctx, in_shape, make_span(axes), true, false);
  TensorListShape<> ref_out_shape = {{
    { 3, 1, 1 },
    { 3, 1, 1 },
    { 1, 1, 1 }
  }};
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);
  EXPECT_GT(sum.GetNumStages(), 1);  // should be more than 1 stage for this scale of reduction
  auto &input_stage = sum.GetStage(0);
  auto &output_stage = sum.GetStage(sum.GetNumStages() - 1);
  for (int i = 0; i < in_shape.num_samples(); i++) {
    auto sample_shape = in_shape[i];
    EXPECT_EQ(input_stage.shape[i].inner, 1);
    EXPECT_EQ(input_stage.shape[i].outer, sample_shape[0]);
    EXPECT_EQ(input_stage.shape[i].reduced_in, sample_shape[1] * sample_shape[2]);
    EXPECT_EQ(output_stage.shape[i].reduced_out, 1);
  }

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<uint8_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 255);

  TestTensorList<int64_t> out;
  out.reshape(req.output_shapes[0]);
  sum.Run(ctx, out.gpu(), in.gpu());

  auto out_cpu = out.cpu(ctx.gpu.stream);
  TestTensorList<int64_t> ref;
  ref.reshape(ref_out_shape);
  auto ref_cpu = ref.cpu();
  for (int i = 0; i < ref_cpu.num_samples(); i++) {
    RefReduce(ref_cpu[i], in_cpu[i], make_span(axes), reductions::sum());
  }

  Check(out_cpu, ref_cpu);
}


TEST(SumImplGPU, Outer) {
  TensorListShape<> in_shape = {{
    { 480, 640, 3 },
    { 720, 1280, 3 },
    { 576, 720, 1 }
  }};
  int axes[] = { 0, 1 };
  SumImplGPU<int64_t, uint8_t> sum;
  KernelContext ctx = {};
  auto req = sum.Setup(ctx, in_shape, make_span(axes), true, false);
  TensorListShape<> ref_out_shape = {{
    { 1, 1, 3 },
    { 1, 1, 3 },
    { 1, 1, 1 }
  }};
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);
  EXPECT_GT(sum.GetNumStages(), 1);  // should be more than 1 stage for this scale of reduction
  auto &input_stage = sum.GetStage(0);
  auto &output_stage = sum.GetStage(sum.GetNumStages() - 1);
  for (int i = 0; i < in_shape.num_samples(); i++) {
    auto sample_shape = in_shape[i];
    EXPECT_EQ(input_stage.shape[i].outer, 1);
    EXPECT_EQ(input_stage.shape[i].inner, sample_shape[2]);
    EXPECT_EQ(input_stage.shape[i].reduced_in, sample_shape[0] * sample_shape[1]);
    EXPECT_EQ(output_stage.shape[i].reduced_out, 1);
  }

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<uint8_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 255);

  TestTensorList<int64_t> out;
  out.reshape(req.output_shapes[0]);
  sum.Run(ctx, out.gpu(), in.gpu());

  auto out_cpu = out.cpu(ctx.gpu.stream);
  TestTensorList<int64_t> ref;
  ref.reshape(ref_out_shape);
  auto ref_cpu = ref.cpu();
  for (int i = 0; i < ref_cpu.num_samples(); i++) {
    RefReduce(ref_cpu[i], in_cpu[i], make_span(axes), reductions::sum());
  }

  Check(out_cpu, ref_cpu);
}

TEST(SumImplGPU, SplitStage) {
  TensorListShape<> in_shape = {{
    { 32, 2, 64000 },
    { 15, 4, 128000 },
    { 72000, 1, 7 }
  }};
  int axes[] = { 0, 2 };
  SumImplGPU<int64_t, uint8_t> sum;
  KernelContext ctx = {};
  auto req = sum.Setup(ctx, in_shape, make_span(axes), true, false);
  TensorListShape<> ref_out_shape = {{
    { 1, 2, 1 },
    { 1, 4, 1 },
    { 1, 1, 1 }
  }};
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);
  EXPECT_GE(sum.GetNumStages(), 4);  // both reduced axes must be split due to large max extent

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<uint8_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 255);

  TestTensorList<int64_t> out;
  out.reshape(req.output_shapes[0]);
  sum.Run(ctx, out.gpu(), in.gpu());

  auto out_cpu = out.cpu(ctx.gpu.stream);
  TestTensorList<int64_t> ref;
  ref.reshape(ref_out_shape);
  auto ref_cpu = ref.cpu();
  for (int i = 0; i < ref_cpu.num_samples(); i++) {
    RefReduce(ref_cpu[i], in_cpu[i], make_span(axes), reductions::sum());
  }

  Check(out_cpu, ref_cpu);
}


TEST(SumImplGPU, WholeSamples) {
  TensorListShape<> in_shape = {{
    { 480, 640 },
    { 640, 480 },
    { 0, 0 },
    { 1280, 300 },
  }};
  int axes[] = { 0, 1 };
  SumImplGPU<int64_t, uint8_t> sum;
  KernelContext ctx = {};
  auto req = sum.Setup(ctx, in_shape, make_span(axes), true, false);
  TensorListShape<> ref_out_shape = {{
    { 1, 1 },
    { 1, 1 },
    { 1, 1 },
    { 1, 1 }
  }};
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);
  EXPECT_GE(sum.GetNumStages(), 2);

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<uint8_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 255);

  TestTensorList<int64_t> out;
  out.reshape(req.output_shapes[0]);
  cudaMemset(out.gpu().data[0], -1, out.gpu().num_elements() * sizeof(int64_t));
  sum.Run(ctx, out.gpu(), in.gpu());

  auto out_cpu = out.cpu(ctx.gpu.stream);
  TestTensorList<int64_t> ref;
  ref.reshape(ref_out_shape);
  auto ref_cpu = ref.cpu();
  for (int i = 0; i < ref_cpu.num_samples(); i++) {
    RefReduce(ref_cpu[i], in_cpu[i], make_span(axes), reductions::sum());
  }

  Check(out_cpu, ref_cpu);
}


TEST(SumImplGPU, All) {
  TensorListShape<> in_shape = {{
    { 480, 640 },
    { 640, 480 },
    { 1280, 300 }
  }};
  int axes[] = { 0, 1 };
  SumImplGPU<int64_t, uint8_t> sum;
  KernelContext ctx = {};
  auto req = sum.Setup(ctx, in_shape, make_span(axes), true, true);
  TensorListShape<> ref_out_shape = {{
    { 1, 1 }
  }};
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);
  EXPECT_GE(sum.GetNumStages(), 2);

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<uint8_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 255);

  TestTensorList<int64_t> out;
  out.reshape(req.output_shapes[0]);
  sum.Run(ctx, out.gpu(), in.gpu());

  auto out_cpu = out.cpu(ctx.gpu.stream);
  int64_t ref = 0;
  for (int i = 0; i < in_cpu.num_samples(); i++) {
    auto tv = in_cpu[i];
    int64_t total_n = tv.num_elements();
    for (int j = 0; j < total_n; j++)
      ref += tv.data[j];
  }
  EXPECT_EQ(*out_cpu.data[0], ref);
}

TEST(SumImplGPU, BatchOnly) {
  TensorListShape<> in_shape = {{
    { 480, 640 },
    { 480, 640 },
    { 480, 640 }
  }};
  SumImplGPU<int64_t, uint8_t> sum;
  KernelContext ctx = {};
  auto req = sum.Setup(ctx, in_shape, {}, true, true);
  TensorListShape<> ref_out_shape = {{
    { 480, 640 }
  }};
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);
  EXPECT_EQ(sum.GetNumStages(), 1);

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<uint8_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 255);

  TestTensorList<int64_t> out;
  out.reshape(req.output_shapes[0]);
  sum.Run(ctx, out.gpu(), in.gpu());

  TestTensorList<int64_t> ref;
  ref.reshape(ref_out_shape);
  auto ref_cpu = ref.cpu();
  int64_t n = ref_cpu.num_elements();
  int64_t N = in_cpu.num_samples();
  for (int j = 0; j < n; j++) {
    int64_t sum = 0;
    for (int i = 0; i < N; i++) {
      sum += in_cpu.data[i][j];
    }
    ref_cpu.data[0][j] = sum;
  }

  auto out_cpu = out.cpu();

  Check(out_cpu, ref_cpu);
}

TEST(SumImplGPU, SplitStageBatch) {
  TensorListShape<> in_shape = {{
    { 32, 3, 64000 },
    { 15, 3, 128000 },
    { 72000, 3, 7 }
  }};
  int axes[] = { 0, 2 };
  SumImplGPU<int64_t, uint8_t> sum;
  KernelContext ctx = {};
  auto req = sum.Setup(ctx, in_shape, make_span(axes), false, true);
  TensorListShape<> ref_out_shape = {{
    { 3 },
  }};
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);
  EXPECT_GE(sum.GetNumStages(), 4);  // both reduced axes must be split due to large max extent

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<uint8_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 255);

  TestTensorList<int64_t> out;
  out.reshape(req.output_shapes[0]);
  sum.Run(ctx, out.gpu(), in.gpu());

  auto out_cpu = out.cpu(ctx.gpu.stream);
  TestTensorList<int64_t> ref_samples, ref;
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
  for (int j = 0; j < n; j++) {
    int64_t sum = 0;
    for (int i = 0; i < N; i++)
      sum += ref_samples_cpu.data[i][j];
    ref_cpu.data[0][j] = sum;
  }

  Check(out_cpu, ref_cpu);
}


TEST(SumImplGPU, All_ZeroSize) {
  TensorListShape<> in_shape = {{
    TensorShape<>{0, 0},
    TensorShape<>{0, 0},
    TensorShape<>{0, 0},
  }};
  int axes[] = { 0, 1 };
  SumImplGPU<int64_t, uint8_t> sum;
  KernelContext ctx = {};
  auto req = sum.Setup(ctx, in_shape, make_span(axes), false, true);
  TensorListShape<> ref_out_shape = {{
    { 1 }
  }};
  EXPECT_EQ(req.output_shapes[0], ref_out_shape);

  ScratchpadAllocator sa;
  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  std::mt19937_64 rng(12345);

  TestTensorList<uint8_t> in;
  in.reshape(in_shape);
  auto in_cpu = in.cpu();
  UniformRandomFill(in_cpu, rng, 0, 255);

  TestTensorList<int64_t> out;
  out.reshape(req.output_shapes[0]);
  cudaMemset(out.gpu().data[0], -1, out.gpu().num_elements() * sizeof(int64_t));
  sum.Run(ctx, out.gpu(), in.gpu());

  EXPECT_EQ(*out.cpu().data[0], 0uL);
}

}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali
