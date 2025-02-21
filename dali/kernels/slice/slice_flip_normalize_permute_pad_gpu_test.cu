// Copyright (c) 2019, 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/slice/slice_kernel_test.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_gpu.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_kernel_test.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace kernels {

template <typename TestArgs>
class SliceFlipNormalizePermutePadGpuTest : public SliceFlipNormalizePermutePadTest<TestArgs> {
 public:
  using InputType = typename TestArgs::InputType;
  using OutputType = typename TestArgs::OutputType;
  static constexpr int Dims = TestArgs::Dims;
  static constexpr int NumSamples = TestArgs::NumSamples;
  static constexpr int DimSize = TestArgs::DimSize;
  using ArgsGenerator = typename TestArgs::ArgsGenerator;
  using KernelType = SliceFlipNormalizePermutePadGpu<OutputType, InputType, Dims>;

  void Run() override {
    KernelContext ctx;
    ctx.gpu.stream = 0;

    TestTensorList<InputType, Dims> test_data;
    this->PrepareData(test_data);

    auto args = this->GenerateArgs(test_data.cpu());

    KernelType kernel;
    KernelRequirements req = kernel.Setup(ctx, test_data.gpu(), args);

    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;

    TensorListShape<> output_shapes = req.output_shapes[0];
    for (int i = 0; i < output_shapes.size(); i++) {
      auto out_shape = args[i].shape;
      auto expected_shape = out_shape;
      for (int d = 0; d < Dims; d++) {
        int perm_d = args[i].permuted_dims[d];
        expected_shape[d] = out_shape[perm_d];
      }
      AssertExpectedDimensions(output_shapes[i], expected_shape);
    }

    TestTensorList<OutputType, Dims> output_data;
    output_data.reshape(output_shapes.to_static<Dims>());
    OutListGPU<OutputType, Dims> out_tlv = output_data.gpu();

    kernel.Run(ctx, out_tlv, test_data.gpu(), args);

    TestTensorList<OutputType, Dims> expected_output;
    this->PrepareExpectedOutput(test_data, args, expected_output);

    EXPECT_NO_FATAL_FAILURE(Check(output_data.cpu(), expected_output.cpu(), EqualEps(1e-6)));
  }
};

TYPED_TEST_SUITE(SliceFlipNormalizePermutePadGpuTest, SLICE_FLIP_NORMALIZE_PERMUTE_TEST_TYPES);

TYPED_TEST(SliceFlipNormalizePermutePadGpuTest, All) {
  this->Run();
}



}  // namespace kernels
}  // namespace dali
