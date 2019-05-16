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

#ifndef DALI_KERNELS_SLICE_SLICE_KERNEL_TEST_H_
#define DALI_KERNELS_SLICE_SLICE_KERNEL_TEST_H_

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "dali/kernels/slice/slice_gpu.h"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/test/test_tensors.h"

#define DEBUG_ENABLED 0
#define DEBUG_OUTPUT \
  if (!DEBUG_ENABLED) ; \
  else std::cout // NOLINT

namespace dali {
namespace kernels {

template <typename Container>
std::string BatchToStr(const Container& batch, const std::string sample_prefix = "Sample ") {
  std::stringstream ss;
  for (int i = 0; i < batch.num_samples(); i++) {
    ss << sample_prefix << i << ":";
    for (auto& x : make_span(batch[i].data, batch[i].num_elements()))
      ss << " " << static_cast<int64_t>(x);
  }
  return ss.str();
}

template <typename InputType_, typename OutputType_, std::size_t Dims_, std::size_t NumSamples_,
          std::size_t DimSize_, typename SliceParamsGenerator_>
struct SliceGPUTestArgs {
  using InputType = InputType_;
  using OutputType = OutputType_;
  static constexpr std::size_t Dims = Dims_;
  static constexpr std::size_t NumSamples = NumSamples_;
  static constexpr std::size_t DimSize = DimSize_;
  using SliceArgsGenerator = SliceParamsGenerator_;
};

template <typename ExpectedShape>
void AssertExpectedDimensions(const TensorShape<>& tensor_shape,
                              const ExpectedShape& expected_shape) {
  for (int j = 0; j < tensor_shape.size(); j++) {
    ASSERT_EQ(expected_shape[j], tensor_shape[j]);
  }
}

template <typename TestArgs>
class SliceGPUTest : public ::testing::Test {
 public:
  using InputType = typename TestArgs::InputType;
  using OutputType = typename TestArgs::OutputType;
  static constexpr std::size_t Dims = TestArgs::Dims;
  static constexpr std::size_t NumSamples = TestArgs::NumSamples;
  static constexpr std::size_t DimSize = TestArgs::DimSize;
  using SliceArgsGenerator = typename TestArgs::SliceArgsGenerator;

  void PrepareExpectedOutut(TestTensorList<InputType, Dims>& input_data,
                            std::vector<SliceArgs<Dims>>& slice_args,
                            TestTensorList<OutputType, Dims>& output_data) {
    auto in = input_data.cpu();
    std::vector<TensorShape<Dims>> output_shapes;
    for (int i = 0; i < in.size(); i++) {
      auto in_sample_shape = in.tensor_shape(i);
      TensorShape<Dims> out_sample_shape(slice_args[i].shape);
      auto& anchor = slice_args[i].anchor;

      for (std::size_t d = 0; d < Dims; d++) {
        ASSERT_TRUE(anchor[d] >= 0 && (anchor[d] + out_sample_shape[d]) <= in_sample_shape[d]);
      }

      output_shapes.push_back(out_sample_shape);
    }
    output_data.reshape(output_shapes);
    auto out = output_data.cpu();

    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      const auto out_shape = out.tensor_shape(i);
      const auto& anchor = slice_args[i].anchor;
      const auto *in_tensor = in.tensor_data(i);
      auto *out_tensor = out.tensor_data(i);

      std::array<int64_t, Dims> in_strides = GetStrides<Dims>(in_shape);
      std::array<int64_t, Dims> out_strides = GetStrides<Dims>(out_shape);

      std::size_t total_size = volume(out_shape);
      for (std::size_t out_idx = 0; out_idx < total_size; out_idx++) {
        std::size_t idx = out_idx;
        std::size_t in_idx = 0;
        for (std::size_t d = 0; d < Dims; d++) {
          std::size_t i_d = idx / out_strides[d];
          idx = idx % out_strides[d];
          in_idx += (anchor[d] + i_d) * in_strides[d];
        }
        out_tensor[out_idx] = in_tensor[in_idx];
      }
    }
  }

  void PrepareData(TestTensorList<InputType, Dims>& test_data) {
    std::vector<int> sample_dims(Dims, DimSize);
    TensorListShape<Dims> shape = uniform_list_shape<Dims>(NumSamples, sample_dims);
    test_data.reshape(shape);

    InputType num = 0;
    auto seq_gen = [&num]() { return num++; };
    Fill(test_data.cpu(), seq_gen);
  }

  std::vector<SliceArgs<Dims>> GenerateSliceArgs(const InListCPU<InputType, Dims>& input_tlv) {
    SliceArgsGenerator generator;
    std::vector<SliceArgs<Dims>> slice_args;
    for (std::size_t i = 0; i < NumSamples; i++) {
      auto shape = input_tlv.tensor_shape(i);
      slice_args.push_back(generator.Get(shape));
    }
    return slice_args;
  }

  void Run() {
    KernelContext ctx;

    TestTensorList<InputType, Dims> test_data;
    PrepareData(test_data);

    DEBUG_OUTPUT << BatchToStr(test_data.cpu(), "Input sample ") << std::endl;

    auto slice_args = GenerateSliceArgs(test_data.cpu());

    SliceGPU<InputType, OutputType, Dims> kernel;
    KernelRequirements kernel_req = kernel.Setup(ctx, test_data.gpu(), slice_args);

    TensorListShape<> output_shapes = kernel_req.output_shapes[0];
    for (int i = 0; i < output_shapes.size(); i++) {
      AssertExpectedDimensions(output_shapes[i], slice_args[i].shape);
    }

    TestTensorList<OutputType, Dims> output_data;
    DEBUG_OUTPUT << "OUTPUT SHAPE " << output_shapes.to_static<Dims>() << std::endl;
    output_data.reshape(output_shapes.to_static<Dims>());
    OutListGPU<OutputType, Dims> out_tlv = output_data.gpu();

    kernel.Run(ctx, out_tlv, test_data.gpu(), slice_args);

    DEBUG_OUTPUT << BatchToStr(output_data.cpu(), "Output sample ") << std::endl;

    TestTensorList<OutputType, Dims> expected_output;
    PrepareExpectedOutut(test_data, slice_args, expected_output);
    DEBUG_OUTPUT << BatchToStr(expected_output.cpu(), "Expected sample ") << std::endl;

    EXPECT_NO_FATAL_FAILURE(Check(output_data.cpu(), expected_output.cpu()));
  }
};

template <std::size_t Dims>
struct SliceArgsGenerator_WholeTensor {
  SliceArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<Dims> args;
    for (std::size_t d = 0; d < Dims; d++) {
      args.anchor[d] = 0;
      args.shape[d] = input_shape[d];
    }
    return args;
  }
};

template <std::size_t Dims>
struct SliceArgsGenerator_HalfAllDims {
  SliceArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<Dims> args;
    for (std::size_t d = 0; d < Dims; d++) {
      args.anchor[d] = input_shape[d] / 2;
      args.shape[d] = input_shape[d] / 2;
    }
    return args;
  }
};

template <std::size_t Dims, int ExtractDim>
struct SliceArgsGenerator_HalfOneDim {
  SliceArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<Dims> args;
    for (std::size_t d = 0; d < Dims; d++) {
      args.anchor[d] = 0;
      args.shape[d] = (d == ExtractDim) ? input_shape[d] / 2 : input_shape[d];
    }
    return args;
  }
};

template <std::size_t Dims>
struct SliceArgsGenerator_ExtractCenterElement {
  SliceArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<Dims> args;
    for (std::size_t d = 0; d < Dims; d++) {
      args.anchor[d] = input_shape[d] / 2;
      args.shape[d] = 1;
    }
    return args;
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_KERNEL_TEST_H_