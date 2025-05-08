// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/reduce/reduce_test.h"
#include "dali/kernels/reduce/reduce_gpu_test.h"

namespace dali {
namespace kernels {
namespace reduce_impl {

TEST(ReduceImplGPU, ReducedShape_NoOp) {
  TensorListShape<> in = {{
    { 1, 2, 3 },
    { 4, 5, 6 }
  }};
  TensorListShape<> ref = {{
    { 1, 2, 3 },
    { 4, 5, 6 }
  }};
  TensorListShape<> out;
  CalculateReducedShape(out, in, {}, false, false);
  EXPECT_EQ(out, ref);
}

TEST(ReduceImplGPU, ReducedShape_BatchOnly) {
  TensorListShape<> in = {{
    { 3, 4, 5 },
    { 3, 4, 5 }
  }};
  TensorListShape<> ref = {{
    { 3, 4, 5 }
  }};
  TensorListShape<> out;
  CalculateReducedShape(out, in, {}, false, true);
  EXPECT_EQ(out, ref);
}

TEST(ReduceImplGPU, ReducedShape_SomeAxes_KeepDims) {
  TensorListShape<> in = {{
    { 2, 3, 4 },
    { 5, 6, 7 }
  }};
  TensorListShape<> ref = {{
    { 1, 3, 1 },
    { 1, 6, 1 }
  }};
  int axes[] = { 0, 2 };
  TensorListShape<> out;
  CalculateReducedShape(out, in, make_span(axes), true, false);
  EXPECT_EQ(out, ref);
}

TEST(ReduceImplGPU, ReducedShape_SomeAxes_NoKeepDims_Batch) {
  TensorListShape<> in = {{
    { 3, 15, 5, 17 },
    { 3, 16, 5, 18 }
  }};
  TensorListShape<> ref = {{
    { 3, 5 }
  }};
  int axes[] = { 1, 3 };
  TensorListShape<> out;
  CalculateReducedShape(out, in, make_span(axes), false, true);
  EXPECT_EQ(out, ref);
}

TEST(ReduceImplGPU, ReducedShape_AllAxes_NoKeepDims) {
  TensorListShape<> in = {{
    { 1, 2 },
    { 3, 4 }
  }};
  TensorListShape<> ref = {{
    TensorShape<>{},
    TensorShape<>{}
  }};
  int axes[] = { 0, 1 };
  TensorListShape<> out;
  CalculateReducedShape(out, in, make_span(axes), false, false);
  EXPECT_EQ(out, ref);
}

TEST(ReduceImplGPU, ReducedShape_AllAxes_KeepDims_Batch) {
  TensorListShape<> in = {{
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 }
  }};
  TensorListShape<> ref = {{
    { 1, 1, 1, 1 }
  }};
  int axes[] = { 0, 1, 2, 3 };
  TensorListShape<> out;
  CalculateReducedShape(out, in, make_span(axes), true, true);
  EXPECT_EQ(out, ref);
}

TEST(ReduceImplGPU, ReducedShape_ScalarInput_batch) {
  TensorListShape<> in = {{
    TensorShape<>{},
    TensorShape<>{}
  }};
  TensorListShape<> ref = {{
    TensorShape<>{}
  }};
  TensorListShape<> out;
  CalculateReducedShape(out, in, {}, true, true);
  EXPECT_EQ(out, ref);
}

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
      if (mask & (1u << a))
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
  TensorListShape<> ref_out_shape = {{
    { 3, 1, 1 },
    { 3, 1, 1 },
    { 1, 1, 1 }
  }};
  int axes[] = { 1, 2 };

  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), true, false);
  EXPECT_GT(test.kernel.GetNumStages(), 1);  // should be more than 1 stage - large extent
  auto &input_stage = test.kernel.GetStage(0);
  auto &output_stage = test.kernel.GetStage(test.kernel.GetNumStages() - 1);
  for (int i = 0; i < in_shape.num_samples(); i++) {
    auto sample_shape = in_shape[i];
    EXPECT_EQ(input_stage.shape[i].inner, 1);
    EXPECT_EQ(input_stage.shape[i].outer, sample_shape[0]);
    EXPECT_EQ(input_stage.shape[i].reduced_in, sample_shape[1] * sample_shape[2]);
    EXPECT_EQ(output_stage.shape[i].reduced_out, 1);
  }

  test.FillData(0, 255);
  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), true, false, reductions::sum());

  test.Check();
}


TEST(SumImplGPU, Outer) {
  TensorListShape<> in_shape = {{
    { 480, 640, 3 },
    { 720, 1280, 3 },
    { 576, 720, 1 }
  }};
  TensorListShape<> ref_out_shape = {{
    { 1, 1, 3 },
    { 1, 1, 3 },
    { 1, 1, 1 }
  }};
  int axes[] = { 0, 1 };

  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), true, false);

  EXPECT_GT(test.kernel.GetNumStages(), 1);  // should be more than 1 stage - large extent
  auto &input_stage = test.kernel.GetStage(0);
  auto &output_stage = test.kernel.GetStage(test.kernel.GetNumStages() - 1);
  for (int i = 0; i < in_shape.num_samples(); i++) {
    auto sample_shape = in_shape[i];
    EXPECT_EQ(input_stage.shape[i].outer, 1);
    EXPECT_EQ(input_stage.shape[i].inner, sample_shape[2]);
    EXPECT_EQ(input_stage.shape[i].reduced_in, sample_shape[0] * sample_shape[1]);
    EXPECT_EQ(output_stage.shape[i].reduced_out, 1);
  }

  test.FillData(0, 255);
  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), true, false, reductions::sum());

  test.Check();
}

TEST(SumImplGPU, SplitStage) {
  TensorListShape<> in_shape = {{
    { 32, 2, 64000 },
    { 15, 4, 128000 },
    { 72000, 1, 7 }
  }};
  TensorListShape<> ref_out_shape = {{
    { 1, 2, 1 },
    { 1, 4, 1 },
    { 1, 1, 1 }
  }};
  int axes[] = { 0, 2 };

  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), true, false);
  EXPECT_GE(test.kernel.GetNumStages(), 4);  // both reduced axes must be split
  test.FillData(0, 255);
  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), true, false, reductions::sum());

  test.Check();
}


TEST(SumImplGPU, WholeSamples) {
  TensorListShape<> in_shape = {{
    { 480, 640 },
    { 640, 480 },
    { 0, 0 },
    { 1280, 300 },
  }};
  TensorListShape<> ref_out_shape = {{
    { 1, 1 },
    { 1, 1 },
    { 1, 1 },
    { 1, 1 }
  }};
  int axes[] = { 0, 1 };
  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), true, false);
  EXPECT_GE(test.kernel.GetNumStages(), 2);  // at least two stages - large extent
  test.FillData(0, 255);
  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), true, false, reductions::sum());

  test.Check();
}


TEST(SumImplGPU, All) {
  TensorListShape<> in_shape = {{
    { 480, 640 },
    { 640, 480 },
    { 1280, 300 }
  }};
  int axes[] = { 0, 1 };
  TensorListShape<> ref_out_shape = {{
    { 1, 1 }
  }};

  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), true, true);
  EXPECT_GE(test.kernel.GetNumStages(), 2);  // at least two stages - sample + all
  test.FillData(0, 255);
  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), true, true, reductions::sum());

  test.Check();
}

TEST(SumImplGPU, BatchOnly) {
  TensorListShape<> in_shape = {{
    { 480, 640 },
    { 480, 640 },
    { 480, 640 }
  }};
  TensorListShape<> ref_out_shape = {{
    { 480, 640 }
  }};

  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, {}, false, true);
  EXPECT_EQ(test.kernel.GetNumStages(), 1);  // just one stage - fold
  test.FillData(0, 255);
  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), {}, false, true, reductions::sum());

  test.Check();
}

TEST(SumImplGPU, SplitStageBatch) {
  TensorListShape<> in_shape = {{
    { 32, 3, 64000 },
    { 15, 3, 128000 },
    { 72000, 3, 7 }
  }};
  TensorListShape<> ref_out_shape = {{
    TensorShape<>{3}
  }};
  int axes[] = { 0, 2 };
  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), false, true);
  test.FillData(0, 255);

  EXPECT_GE(test.kernel.GetNumStages(), 4);  // both reduced axes must be split
  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), false, true, reductions::sum());

  test.Check();
}


TEST(SumImplGPU, All_ZeroSize) {
  TensorListShape<> in_shape = {{
    TensorShape<>{0, 0},
    TensorShape<>{0, 0},
    TensorShape<>{0, 0},
  }};
  TensorListShape<> ref_out_shape = {{
    TensorShape<>{}
  }};
  int axes[] = { 0, 1 };

  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), false, true);
  test.FillData(0, 255);
  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), false, true, reductions::sum());

  test.Check();
}

TEST(SumImplGPU, Partial_ZeroShape_Middle) {
  TensorListShape<> in_shape = {{
    { 3, 0, 5 },
    { 6, 0, 8 },
    { 9, 0, 2 }
  }};
  TensorListShape<> ref_out_shape = {{
    TensorShape<>{3, 5},
    TensorShape<>{6, 8},
    TensorShape<>{9, 2}
  }};
  int axes[] = { 1 };
  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), false, false);
  test.FillData(0, 255);

  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), false, false, reductions::sum());

  test.Check();
}

TEST(SumImplGPU, Partial_ZeroShape_Inner) {
  TensorListShape<> in_shape = {{
    { 3, 0, 0 },
    { 6, 0, 0 },
    { 9, 0, 0 }
  }};
  TensorListShape<> ref_out_shape = {{
    TensorShape<>{3},
    TensorShape<>{6},
    TensorShape<>{9}
  }};
  int axes[] = { 1, 2 };
  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), false, false);
  test.FillData(0, 255);

  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), false, false, reductions::sum());

  test.Check();
}

TEST(SumImplGPU, Partial_ZeroShape_Outer) {
  TensorListShape<> in_shape = {{
    { 0, 4, 5 },
    { 0, 7, 8 },
    { 0, 1, 2 }
  }};
  TensorListShape<> ref_out_shape = {{
    TensorShape<>{4, 5},
    TensorShape<>{7, 8},
    TensorShape<>{1, 2}
  }};
  int axes[] = { 0 };
  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), false, false);
  test.FillData(0, 255);

  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), false, false, reductions::sum());

  test.Check();
}

TEST(SumImplGPU, Partial_ZeroShape_OuterInner) {
  TensorListShape<> in_shape = {{
    { 0, 4, 0 },
    { 0, 7, 0 },
    { 0, 1, 0 }
  }};
  TensorListShape<> ref_out_shape = {{
    TensorShape<>{4},
    TensorShape<>{7},
    TensorShape<>{1}
  }};
  int axes[] = { 0, 2 };
  testing::ReductionKernelTest<SumImplGPU<int64_t, uint8_t>, int64_t, uint8_t> test;
  test.Setup(in_shape, ref_out_shape, make_span(axes), false, false);
  test.FillData(0, 255);

  test.Run();

  RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), false, false, reductions::sum());

  test.Check();
}

}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali
