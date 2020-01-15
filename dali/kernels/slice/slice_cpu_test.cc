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
  static constexpr int Dims = TestArgs::Dims;
  static constexpr int NumSamples = TestArgs::NumSamples;
  static constexpr int DimSize = TestArgs::DimSize;
  using ArgsGenerator = typename TestArgs::ArgsGenerator;
  using KernelType = SliceCPU<OutputType, InputType, Dims>;

  void Run() override {
    KernelContext ctx;

    TestTensorList<InputType, Dims> test_data;
    this->PrepareData(test_data);

    auto test_data_cpu = test_data.cpu();
    auto slice_args = this->GenerateArgs(test_data_cpu);

    TestTensorList<OutputType, Dims> expected_output;
    this->PrepareExpectedOutput(test_data, slice_args, expected_output);

    TensorListShape<> output_shapes(NumSamples, Dims);
    std::vector<KernelType> kernels(NumSamples);
    for (int i = 0; i < NumSamples; i++) {
      auto &kernel = kernels[i];
      KernelRequirements kernel_req = kernel.Setup(ctx, test_data_cpu[i], slice_args[i]);
      TensorShape<Dims> output_shape = kernel_req.output_shapes[0][0].to_static<Dims>();
      AssertExpectedDimensions(output_shape, slice_args[i].shape);
      output_shapes.set_tensor_shape(i, output_shape);
    }
    TestTensorList<OutputType, Dims> output_data;
    output_data.reshape(std::move(output_shapes).to_static<Dims>());
    OutListCPU<OutputType, Dims> out_tlv = output_data.cpu();

    for (int i = 0; i < NumSamples; i++) {
      auto &kernel = kernels[i];
      auto out_tv = out_tlv[i];
      auto in_tv = test_data_cpu[i];
      kernel.Run(ctx, out_tv, in_tv, slice_args[i]);
    }
    EXPECT_NO_FATAL_FAILURE(Check(output_data.cpu(), expected_output.cpu()));
  }
};

TYPED_TEST_SUITE(SliceCPUTest, SLICE_TEST_TYPES);

TYPED_TEST(SliceCPUTest, All) {
  this->Run();
}

template <typename TestArgs>
class SliceCPUTest_CpuOnlyTests : public SliceCPUTest<TestArgs> {};

TYPED_TEST_SUITE(SliceCPUTest_CpuOnlyTests, SLICE_TEST_TYPES_CPU_ONLY);

TYPED_TEST(SliceCPUTest_CpuOnlyTests, All) {
  this->Run();
}

}  // namespace kernels
}  // namespace dali
