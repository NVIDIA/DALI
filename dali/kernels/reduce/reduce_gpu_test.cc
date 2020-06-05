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
#include "dali/kernels/reduce/reduce_gpu_test.h"

namespace dali {
namespace kernels {

TEST(SumGPU, SplitStageBatch) {
  TensorListShape<> in_shape = {{
    { 32, 3, 64000 },
    { 15, 3, 128000 },
    { 72000, 3, 7 }
  }};
  TensorListShape<> ref_out_shape = {{
    TensorShape<>{3}
  }};
  int axes[] = { 0, 2 };

  testing::ReductionKernelTest<SumGPU<uint64_t, uint8_t>, uint64_t, uint8_t> test;
  for (int iter = 0; iter < 3; iter++) {
    test.Setup(in_shape, ref_out_shape, make_span(axes), false, true);
    test.FillData(0, 255);
    test.Run();
    RefReduce(test.ref.cpu(), test.in.cpu(), make_span(axes), false, true, reductions::sum());
    test.Check();
  }
}

}  // namespace kernels
}  // namespace dali
