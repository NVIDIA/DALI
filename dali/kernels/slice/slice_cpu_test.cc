// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/slice/slice_cpu.h"

namespace dali {
namespace kernels {

template <typename TestArgs>
class SliceCPUTest : public SliceTest<TestArgs> {
 public:
  using InputType = typename TestArgs::InputType;
  using OutputType = typename TestArgs::OutputType;
  static constexpr std::size_t Dims = TestArgs::Dims;
  static constexpr std::size_t NumSamples = TestArgs::NumSamples;
  static constexpr std::size_t DimSize = TestArgs::DimSize;
  using SliceArgsGenerator = typename TestArgs::SliceArgsGenerator;
  using KernelType = SliceCPU<OutputType, InputType, Dims>;

  void Run() override {
    KernelContext ctx;

    TestTensorList<InputType, Dims> test_data;
    SliceTest<TestArgs>::PrepareData(test_data);

    LOG_LINE << BatchToStr(test_data.cpu(), "Input sample ") << std::endl;

    auto test_data_cpu = test_data.cpu();
    auto slice_args = SliceTest<TestArgs>::GenerateSliceArgs(test_data_cpu);

    TestTensorList<OutputType, Dims> expected_output;
    SliceTest<TestArgs>::PrepareExpectedOutput(test_data, slice_args, expected_output);
    LOG_LINE << BatchToStr(expected_output.cpu(), "Expected sample ") << std::endl;

    TensorListShape<> output_shapes;
    output_shapes.resize(NumSamples, Dims);
    std::vector<KernelType> kernels(NumSamples);
    for (std::size_t i = 0; i < NumSamples; i++) {
      auto &kernel = kernels[i];
      KernelRequirements kernel_req = kernel.Setup(ctx, test_data_cpu[i], slice_args[i]);
      TensorShape<Dims> output_shape = kernel_req.output_shapes[0][0].to_static<Dims>();
      AssertExpectedDimensions(output_shape, slice_args[i].shape);
      output_shapes.set_tensor_shape(i, output_shape);
    }
    TestTensorList<OutputType, Dims> output_data;
    LOG_LINE << "OUTPUT SHAPE " << output_shapes.to_static<Dims>() << std::endl;
    output_data.reshape(output_shapes.to_static<Dims>());
    OutListCPU<OutputType, Dims> out_tlv = output_data.cpu();

    for (std::size_t i = 0; i < NumSamples; i++) {
      auto &kernel = kernels[i];
      auto out_tv = out_tlv[i];
      auto in_tv = test_data_cpu[i];
      kernel.Run(ctx, out_tv, in_tv, slice_args[i]);
    }
    LOG_LINE << BatchToStr(output_data.cpu(), "Output sample ") << std::endl;
    EXPECT_NO_FATAL_FAILURE(Check(output_data.cpu(), expected_output.cpu()));
  }
};

TYPED_TEST_SUITE(SliceCPUTest, SLICE_TEST_TYPES);

TYPED_TEST(SliceCPUTest, All) {
  this->Run();
}

}  // namespace kernels
}  // namespace dali
