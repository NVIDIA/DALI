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
#include "dali/kernels/reduce/reduce_gpu.h"
#include "dali/kernels/reduce/reduce_test.h"
#include "dali/kernels/scratch.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {
namespace kernels {

TEST(SumGPU, SplitStageBatch) {
  TensorListShape<> in_shape = {{
    { 32, 3, 64000 },
    { 15, 3, 128000 },
    { 72000, 3, 7 }
  }};
  int axes[] = { 0, 2 };
  SumGPU<uint64_t, uint8_t> sum;
  KernelContext ctx = {};
  auto req = sum.Setup(ctx, in_shape, make_span(axes), false, true);
  TensorListShape<> ref_out_shape = {{
    TensorShape<>{3}
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

  TestTensorList<uint64_t> out;
  out.reshape(req.output_shapes[0]);
  sum.Run(ctx, out.gpu(), in.gpu());

  auto out_cpu = out.cpu(ctx.gpu.stream);
  TestTensorList<uint64_t> ref_samples, ref;
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

}  // namespace kernels
}  // namespace dali
