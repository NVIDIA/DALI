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
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/slice/slice_kernel_utils.h"

namespace dali {
namespace kernels {

template <typename InputType_, typename OutputType_, int Dims_, int NumSamples_,
          int DimSize_, typename ParamsGenerator_, int DimSize0_ = DimSize_,
          int DimSize1_ = DimSize_, int DimSize2_ = DimSize_>
struct SliceTestArgs {
  using InputType = InputType_;
  using OutputType = OutputType_;
  static constexpr int Dims = Dims_;
  static constexpr int NumSamples = NumSamples_;
  static constexpr int DimSize = DimSize_;
  static constexpr int DimSize0 = DimSize0_;
  static constexpr int DimSize1 = DimSize1_;
  static constexpr int DimSize2 = DimSize2_;
  using ArgsGenerator = ParamsGenerator_;
};

template <typename ExpectedShape>
void AssertExpectedDimensions(const TensorShape<>& tensor_shape,
                              const ExpectedShape& expected_shape) {
  for (int j = 0; j < tensor_shape.size(); j++) {
    ASSERT_EQ(expected_shape[j], tensor_shape[j]);
  }
}

template <typename TestArgs>
class SliceTest : public ::testing::Test {
 public:
  using InputType = typename TestArgs::InputType;
  using OutputType = typename TestArgs::OutputType;
  static constexpr int Dims = TestArgs::Dims;
  static constexpr int NumSamples = TestArgs::NumSamples;
  static constexpr int DimSize = TestArgs::DimSize;
  static constexpr int DimSize0 = TestArgs::DimSize0;
  static constexpr int DimSize1 = TestArgs::DimSize1;
  static constexpr int DimSize2 = TestArgs::DimSize2;
  using ArgsGenerator = typename TestArgs::ArgsGenerator;

  void PrepareData(TestTensorList<InputType, Dims>& test_data) {
    std::vector<int> sample_dims(Dims, static_cast<int>(DimSize));
    sample_dims[0] = DimSize0;
    if (Dims > 1) {
      sample_dims[1] = DimSize1;
    }
    if (Dims > 2) {
      sample_dims[2] = DimSize2;
    }
    TensorListShape<Dims> shape = uniform_list_shape<Dims>(NumSamples, sample_dims);
    test_data.reshape(shape);

    InputType num = 0;
    auto seq_gen = [&num]() { return num++; };
    Fill(test_data.cpu(), seq_gen);
  }

  void PrepareExpectedOutput(TestTensorList<InputType, Dims>& input_data,
                             std::vector<SliceArgs<OutputType, Dims>>& slice_args,
                             TestTensorList<OutputType, Dims>& output_data) {
    auto in = input_data.cpu();
    std::vector<TensorShape<Dims>> output_shapes;
    for (int i = 0; i < in.size(); i++) {
      TensorShape<Dims> out_sample_shape(slice_args[i].shape);
      auto& anchor = slice_args[i].anchor;
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

      int channel_dim = slice_args[i].channel_dim;
      auto *fill_values = slice_args[i].fill_values.data();
      int fill_values_size = slice_args[i].fill_values.size();
      if (fill_values_size > 1) {
        ASSERT_GE(channel_dim, 0);
        ASSERT_EQ(fill_values_size, out_shape[channel_dim]);
      }

      auto in_strides = GetStrides(in_shape);
      auto out_strides = GetStrides(out_shape);

      int64_t total_size = volume(out_shape);

      int i_c = 0;
      for (int64_t out_idx = 0; out_idx < total_size; out_idx++) {
        int64_t idx = out_idx;
        int64_t in_idx = 0;
        bool out_of_bounds = false;
        for (int d = 0; d < Dims; d++) {
          int i_d = idx / out_strides[d];
          idx = idx % out_strides[d];

          if (d == channel_dim)
            i_c = i_d;
          out_of_bounds |= anchor[d] + i_d < 0;
          out_of_bounds |= anchor[d] + i_d >= in_shape[d];
          if (!out_of_bounds) {
            in_idx += (anchor[d] + i_d) * in_strides[d];
          }
        }

        assert(i_c >= 0 && i_c < fill_values_size);
        out_tensor[out_idx] =
            out_of_bounds ? fill_values[i_c] : Convert<OutputType>(in_tensor[in_idx]);
      }
    }
  }

  std::vector<SliceArgs<OutputType, Dims>> GenerateArgs(
      const InListCPU<InputType, Dims>& input_tlv) {
    ArgsGenerator generator;
    std::vector<SliceArgs<OutputType, Dims>> slice_args;
    for (int i = 0; i < NumSamples; i++) {
      auto shape = input_tlv.tensor_shape(i);
      slice_args.push_back(generator.Get(shape));
    }
    return slice_args;
  }

  virtual void Run() = 0;
};

template <typename OutputType, int Dims>
struct ArgsGen_WholeTensor {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, Dims> args;
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = 0;
      args.shape[d] = input_shape[d];
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct ArgsGen_HalfAllDims {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, Dims> args;
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = input_shape[d] / 2;
      args.shape[d] = input_shape[d] / 2;
    }
    return args;
  }
};

template <typename OutputType, int Dims, int ExtractDim>
struct ArgsGen_HalfOneDim {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, Dims> args;
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = 0;
      args.shape[d] = (d == ExtractDim) ? input_shape[d] / 2 : input_shape[d];
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct ArgsGen_ExtractCenterElement {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, Dims> args;
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = input_shape[d] / 2;
      args.shape[d] = 1;
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct ArgsGen_BiggerThanInputSlice {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, Dims> args;
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = -input_shape[d] / 2;
      args.shape[d] = input_shape[d] * 2;
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct ArgsGen_LeftSideOutOfBounds{
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, Dims> args;
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = -input_shape[d] / 2;
      args.shape[d] = input_shape[d];
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct ArgsGen_RightSideOutOfBounds{
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, Dims> args;
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = input_shape[d] / 2;
      args.shape[d] = input_shape[d];
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct ArgsGen_CompletelyOutOfBounds{
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, Dims> args;
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = 4 * input_shape[d];
      args.shape[d] = input_shape[d] / 2;
    }
    return args;
  }
};

template <typename OutputType, int Dims = 3>
struct ArgsGen_SingleValuePad {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, 3> args;
    args.anchor[0] = -input_shape[0] / 2;
    args.anchor[1] = -input_shape[1] / 2;
    args.anchor[2] = 0;
    args.shape[0] = 2 * input_shape[0];
    args.shape[1] = 2 * input_shape[1];
    args.shape[2] = input_shape[2];
    args.fill_values = {128};
    args.channel_dim = -1;
    return args;
  }
};

template <typename OutputType, int Dims = 3>
struct ArgsGen_MultiChannelPad {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, 3> args;
    args.anchor[0] = -input_shape[0] / 2;
    args.anchor[1] = -input_shape[1] / 2;
    args.anchor[2] = 0;
    args.shape[0] = 2 * input_shape[0];
    args.shape[1] = 2 * input_shape[1];
    args.shape[2] = input_shape[2];
    args.fill_values = {100, 110, 120};
    args.channel_dim = 2;
    return args;
  }
};


template <typename OutputType, int Dims = 3>
struct ArgsGen_MultiChannelPad_ChFirst {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, 3> args;
    args.anchor[0] = 0;
    args.anchor[1] = -input_shape[1] / 2;
    args.anchor[2] = -input_shape[2] / 2;
    args.shape[0] = input_shape[0];
    args.shape[1] = 2 * input_shape[1];
    args.shape[2] = 2 * input_shape[2];
    args.fill_values = {100, 110, 120};
    args.channel_dim = 0;
    return args;
  }
};

template <typename OutputType, int Dims = 3>
struct ArgsGen_PadAlsoChDim {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, 3> args;
    args.anchor[0] = -input_shape[0] / 2;
    args.anchor[1] = -input_shape[1] / 2;
    args.anchor[2] = 0;
    args.shape[0] = 2 * input_shape[0];
    args.shape[1] = 2 * input_shape[1];
    args.shape[2] = input_shape[2] + 1;
    args.fill_values = {100};
    args.channel_dim = -1;
    return args;
  }
};

template <typename OutputType, int Dims = 3>
struct ArgsGen_PadAlsoChDim_MultiChannelFillValues {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, 3> args;
    args.anchor[0] = -input_shape[0] / 2;
    args.anchor[1] = -input_shape[1] / 2;
    args.anchor[2] = 0;
    args.shape[0] = 2 * input_shape[0];
    args.shape[1] = 2 * input_shape[1];
    args.shape[2] = input_shape[2] + 1;
    args.fill_values = {100, 110, 120, 128};
    args.channel_dim = 2;
    return args;
  }
};

template <typename OutputType, int Dims = 3>
struct ArgsGen_PadOnlyChDim {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, 3> args;
    args.anchor[0] = 0;
    args.anchor[1] = 0;
    args.anchor[2] = 0;
    args.shape[0] = input_shape[0];
    args.shape[1] = input_shape[1];
    args.shape[2] = input_shape[2] + 1;
    args.fill_values = {100};
    args.channel_dim = -1;
    return args;
  }
};

template <typename OutputType, int Dims = 3>
struct ArgsGen_PadAlsoChDim_ChFirst {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, 3> args;
    args.anchor[0] = 0;
    args.anchor[1] = -input_shape[1] / 2;
    args.anchor[2] = -input_shape[2] / 2;
    args.shape[0] = input_shape[0] + 1;
    args.shape[1] = 2 * input_shape[1];
    args.shape[2] = 2 * input_shape[2];
    args.fill_values = {100};
    args.channel_dim = -1;
    return args;
  }
};

template <typename OutputType, int Dims = 3>
struct ArgsGen_PadAlsoChDim_ChFirst_MultiChannelFillValues {
  SliceArgs<OutputType, Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceArgs<OutputType, 3> args;
    args.anchor[0] = 0;
    args.anchor[1] = -input_shape[1] / 2;
    args.anchor[2] = -input_shape[2] / 2;
    args.shape[0] = input_shape[0] + 1;
    args.shape[1] = 2 * input_shape[1];
    args.shape[2] = 2 * input_shape[2];
    args.fill_values = {100, 110, 120, 128};
    args.channel_dim = 0;
    return args;
  }
};

using SLICE_TEST_TYPES = ::testing::Types<
    SliceTestArgs<int, int, 3, 1, 2, ArgsGen_WholeTensor<int, 3>>,
    SliceTestArgs<int, int, 2, 1, 6, ArgsGen_HalfAllDims<int, 2>>,
    SliceTestArgs<int, int, 4, 1, 2, ArgsGen_HalfAllDims<int, 4>>,
    SliceTestArgs<int, int, 3, 1, 2, ArgsGen_HalfOneDim<int, 3, 0>>,
    SliceTestArgs<int, int, 3, 1, 2, ArgsGen_HalfOneDim<int, 3, 1>>,
    SliceTestArgs<int, int, 3, 1, 2, ArgsGen_HalfOneDim<int, 3, 2>>,
    SliceTestArgs<float, float, 3, 1, 2, ArgsGen_HalfOneDim<float, 3, 2>>,
    SliceTestArgs<int, float, 3, 1, 2, ArgsGen_HalfOneDim<float, 3, 2>>,
    SliceTestArgs<float, int, 3, 1, 2, ArgsGen_HalfOneDim<int, 3, 2>>,
    SliceTestArgs<int, int, 3, 10, 2, ArgsGen_HalfOneDim<int, 3, 2>>,
    SliceTestArgs<int, int, 10, 1, 2, ArgsGen_HalfAllDims<int, 10>>,
    SliceTestArgs<uint8_t, uint8_t, 3, 1, 2, ArgsGen_HalfAllDims<uint8_t, 3>>,
    SliceTestArgs<uint8_t, uint8_t, 1, 1, 2, ArgsGen_HalfAllDims<uint8_t, 1>>,
    SliceTestArgs<uint8_t, uint8_t, 2, 1, 1024, ArgsGen_HalfAllDims<uint8_t, 2>>,
    SliceTestArgs<uint8_t, uint8_t, 2, 100, 1024, ArgsGen_HalfAllDims<uint8_t, 2>>,
    SliceTestArgs<uint8_t, uint8_t, 3, 3, 256, ArgsGen_HalfAllDims<uint8_t, 3>>,
    SliceTestArgs<int, int, 2, 1, 3, ArgsGen_ExtractCenterElement<int, 2>>,
    SliceTestArgs<int, int, 1, 1, 20, ArgsGen_BiggerThanInputSlice<int, 1>>,
    SliceTestArgs<int, int, 2, 1, 20, ArgsGen_BiggerThanInputSlice<int, 2>>,
    SliceTestArgs<int, int, 1, 1, 21, ArgsGen_LeftSideOutOfBounds<int, 1>>,
    SliceTestArgs<int, int, 2, 1, 21, ArgsGen_LeftSideOutOfBounds<int, 2>>,
    SliceTestArgs<int, int, 1, 1, 22, ArgsGen_RightSideOutOfBounds<int, 1>>,
    SliceTestArgs<int, int, 2, 1, 22, ArgsGen_RightSideOutOfBounds<int, 2>>,
    SliceTestArgs<int, int, 2, 1, 22, ArgsGen_CompletelyOutOfBounds<int, 2>>,
    SliceTestArgs<int, int, 3, 1, 20, ArgsGen_SingleValuePad<int, 3>, 20, 20, 3>,
    SliceTestArgs<int, int, 3, 1, 20, ArgsGen_MultiChannelPad<int, 3>, 20, 20, 3>,
    SliceTestArgs<int, int, 3, 1, 20, ArgsGen_MultiChannelPad_ChFirst<int, 3>, 3, 20, 20>,
    SliceTestArgs<int, int, 3, 1, 20, ArgsGen_PadAlsoChDim<int, 3>, 20, 20, 3>,
    SliceTestArgs<int, int, 3, 1, 20, ArgsGen_PadAlsoChDim_MultiChannelFillValues<int, 3>, 20, 20, 3>,  // NOLINT
    SliceTestArgs<int, int, 3, 1, 20, ArgsGen_PadOnlyChDim<int, 3>, 20, 20, 3>,
    SliceTestArgs<int, int, 3, 1, 20, ArgsGen_PadAlsoChDim_ChFirst<int, 3>, 3, 20, 20>,
    SliceTestArgs<int, int, 3, 1, 20, ArgsGen_PadAlsoChDim_ChFirst_MultiChannelFillValues<int, 3>, 3, 20, 20>,  // NOLINT
    SliceTestArgs<int, int, 3, 10, 20, ArgsGen_PadAlsoChDim_ChFirst<int, 3>, 3, 20, 20>,
    SliceTestArgs<int, int, 3, 10, 20, ArgsGen_PadAlsoChDim_ChFirst_MultiChannelFillValues<int, 3>, 3, 20, 20>  // NOLINT
>;

using SLICE_TEST_TYPES_CPU_ONLY = ::testing::Types<
    SliceTestArgs<int, float16, 3, 1, 2, ArgsGen_WholeTensor<float16, 3>>,
    SliceTestArgs<float16, int, 3, 1, 2, ArgsGen_WholeTensor<int, 3>>,
    SliceTestArgs<float16, float16, 3, 1, 2, ArgsGen_WholeTensor<float16, 3>>
>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_KERNEL_TEST_H_
