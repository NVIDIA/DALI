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
#include "dali/kernels/slice/slice_flip_normalize_permute_cpu.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_kernel_test.h"

namespace dali {
namespace kernels {

template <typename TestArgs>
class SliceFlipNormalizePermuteCPUTest : public SliceFlipNormalizePermuteTest<TestArgs> {
 public:
  using InputType = typename TestArgs::InputType;
  using OutputType = typename TestArgs::OutputType;
  static constexpr size_t Dims = TestArgs::Dims;
  static constexpr size_t NumSamples = TestArgs::NumSamples;
  static constexpr size_t DimSize = TestArgs::DimSize;
  using ArgsGenerator = typename TestArgs::ArgsGenerator;
  using KernelType = SliceFlipNormalizePermuteCPU<OutputType, InputType, Dims>;

  void Run() override {
    KernelContext ctx;

    TestTensorList<InputType, Dims> test_data;
    this->PrepareData(test_data);

    auto test_data_cpu = test_data.cpu();
    auto args = this->GenerateArgs(test_data_cpu);

    TestTensorList<OutputType, Dims> expected_output;
    this->PrepareExpectedOutput(test_data, args, expected_output);

    TensorListShape<> output_shapes;
    output_shapes.resize(NumSamples, Dims);
    std::vector<KernelType> kernels(NumSamples);
    for (size_t i = 0; i < NumSamples; i++) {
      auto &kernel = kernels[i];
      KernelRequirements kernel_req = kernel.Setup(ctx, test_data_cpu[i], args[i]);
      TensorShape<Dims> output_shape = kernel_req.output_shapes[0][0].to_static<Dims>();

      auto padded_out_shape = args[i].should_pad ? args[i].padded_shape : args[i].shape;
      auto expected_shape = padded_out_shape;
      for (size_t d = 0; d < Dims; d++) {
        size_t perm_d = args[i].should_permute ? args[i].permuted_dims[d] : d;
        expected_shape[d] = padded_out_shape[perm_d];
      }
      AssertExpectedDimensions(output_shape, expected_shape);
      output_shapes.set_tensor_shape(i, output_shape);
    }
    TestTensorList<OutputType, Dims> output_data;
    output_data.reshape(std::move(output_shapes).to_static<Dims>());
    OutListCPU<OutputType, Dims> out_tlv = output_data.cpu();

    for (size_t i = 0; i < NumSamples; i++) {
      auto &kernel = kernels[i];
      auto out_tv = out_tlv[i];
      auto in_tv = test_data_cpu[i];
      kernel.Run(ctx, out_tv, in_tv, args[i]);
    }
    EXPECT_NO_FATAL_FAILURE(Check(output_data.cpu(), expected_output.cpu()));
  }
};

TYPED_TEST_SUITE(SliceFlipNormalizePermuteCPUTest, SLICE_FLIP_NORMALIZE_PERMUTE_TEST_TYPES);

TYPED_TEST(SliceFlipNormalizePermuteCPUTest, All) {
  this->Run();
}

}  // namespace kernels
}  // namespace dali
