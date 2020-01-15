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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_KERNEL_TEST_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_KERNEL_TEST_H_

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "dali/kernels/slice/slice_kernel_test.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"

namespace dali {
namespace kernels {

template <typename TestArgs>
class SliceFlipNormalizePermutePadTest : public ::testing::Test {
 public:
  using InputType = typename TestArgs::InputType;
  using OutputType = typename TestArgs::OutputType;
  static constexpr int Dims = TestArgs::Dims;
  static constexpr int NumSamples = TestArgs::NumSamples;
  static constexpr int DimSize = TestArgs::DimSize;
  static constexpr int DimSize0 = TestArgs::DimSize0;
  static constexpr int DimSize1 = TestArgs::DimSize1;
  using ArgsGenerator = typename TestArgs::ArgsGenerator;
  using KernelArgs = SliceFlipNormalizePermutePadArgs<Dims>;

  void PrepareData(TestTensorList<InputType, Dims>& test_data) {
    std::vector<int> sample_dims(Dims, static_cast<int>(DimSize));
    sample_dims[0] = DimSize0;
    sample_dims[1] = DimSize1;
    TensorListShape<Dims> shape = uniform_list_shape<Dims>(NumSamples, sample_dims);
    test_data.reshape(shape);

    InputType num = 0;
    auto seq_gen = [&num]() { return num++; };
    Fill(test_data.cpu(), seq_gen);
  }

  void PrepareExpectedOutput(TestTensorList<InputType, Dims>& input_data,
                             std::vector<KernelArgs>& args,
                             TestTensorList<OutputType, Dims>& output_data) {
    auto in = input_data.cpu();
    std::vector<TensorShape<Dims>> output_shapes;
    std::vector<TensorShape<Dims>> slice_shapes;
    for (int i = 0; i < in.size(); i++) {
      auto shape = args[i].shape;
      slice_shapes.push_back(shape);
      auto padded_shape = args[i].padded_shape;
      auto out_shape = padded_shape;
      for (int d = 0; d < Dims; d++) {
        auto perm_d = args[i].permuted_dims[d];
        out_shape[d] = padded_shape[perm_d];
      }
      TensorShape<Dims> out_sample_shape(out_shape);
      output_shapes.push_back(out_sample_shape);
    }
    output_data.reshape(output_shapes);
    auto out = output_data.cpu();

    for (int i = 0; i < in.size(); i++) {
      const auto& anchor = args[i].anchor;
      const auto& flip = args[i].flip;
      const auto *in_tensor = in.tensor_data(i);
      auto *out_tensor = out.tensor_data(i);
      const auto &permuted_dims = args[i].permuted_dims;
      auto slice_shape = slice_shapes[i];

      const auto in_shape = in.tensor_shape(i);
      auto in_strides = GetStrides(in_shape);

      const auto out_shape = out.tensor_shape(i);
      auto out_strides = GetStrides(out_shape);

      // normalization
      auto mean = args[i].mean;
      auto inv_stddev = args[i].inv_stddev;
      ASSERT_EQ(mean.size(), inv_stddev.size());

      // Very naive implementation just for test purposes
      size_t total_size = volume(out_shape);
      for (size_t out_idx = 0; out_idx < total_size; out_idx++) {
        size_t idx = out_idx;
        size_t in_idx = 0;
        bool is_zero_pad = false;
        for (int d = 0; d < Dims; d++) {
          auto perm_d = permuted_dims[d];
          int i_d = idx / out_strides[d];
          is_zero_pad = is_zero_pad ||
            (out_shape[d] > slice_shape[perm_d] && i_d >= slice_shape[perm_d]);
          idx = idx % out_strides[d];
          auto offset = flip[perm_d] ?
            (anchor[perm_d] + slice_shape[perm_d] - 1 - i_d) * in_strides[perm_d] :
            (anchor[perm_d] + i_d) * in_strides[perm_d];
          in_idx += offset;
        }

        OutputType output_value = 0;
        if (!is_zero_pad) {
          if (!mean.empty() && !inv_stddev.empty()) {
            auto c = mean.size() == 1 ? 0 : out_idx % out_shape[Dims - 1];
            float fpout = (static_cast<float>(in_tensor[in_idx]) - mean[c]) * inv_stddev[c];
            if (std::is_integral<OutputType>::value) {
              output_value = clamp<OutputType>(std::roundf(fpout));
            } else {
              output_value = clamp<OutputType>(fpout);
            }
          } else {
            if (std::is_integral<OutputType>::value && std::is_floating_point<InputType>::value) {
              output_value = clamp<OutputType>(std::roundf(in_tensor[in_idx]));
            } else {
              output_value = clamp<OutputType>(in_tensor[in_idx]);
            }
          }
        }
        out_tensor[out_idx] = output_value;
      }
    }
  }

  std::vector<KernelArgs> GenerateArgs(const InListCPU<InputType, Dims>& input_tlv) {
    ArgsGenerator generator;
    std::vector<KernelArgs> args;
    for (int i = 0; i < NumSamples; i++) {
      auto shape = input_tlv.tensor_shape(i);
      args.push_back(generator.Get(shape));
    }
    return args;
  }

  virtual void Run() = 0;
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_CopyOnly {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_SliceOnly {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    auto shape = input_shape;
    shape[0] /= 2;
    shape[1] /= 2;
    SliceFlipNormalizePermutePadArgs<Dims> args(shape);
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_SliceOnly_WithAnchor {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    auto shape = input_shape;
    shape[0] = input_shape[0]/2;
    shape[1] = input_shape[0]/2;
    SliceFlipNormalizePermutePadArgs<Dims> args(shape);
    args.anchor[0] = input_shape[0]/2;
    args.anchor[1] = input_shape[1]/2;
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_FlipHW {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    // assuming last dims are HWC, flip H and W
    args.flip[Dims-2] = true;
    args.flip[Dims-3] = true;
    return args;
  }
};

template <typename OutputType, int Dims, int FlipDim>
struct SliceFlipNormPermArgsGen_FlipDim {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    args.flip[FlipDim] = true;
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_NormalizeOnly {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    args.mean.resize(args.shape[Dims-1]);
    args.inv_stddev.resize(args.shape[Dims-1]);
    for (int i = 0; i < args.shape[Dims-1]; i++) {
      args.mean[i] = 3.5f + 0.1f * i;
      args.inv_stddev[i] = 1 / (8.0f + 0.1f * i);
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_NormalizeOnly_Scalar {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    args.mean = { 3.5f };
    args.inv_stddev = { 1.f / 8.0f };
    return args;
  }
};

template <typename OutputType, int Dims, int FlipDim>
struct SliceFlipNormPermArgsGen_NormalizeAndFlipDim {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    args.flip[FlipDim] = true;
    args.mean.resize(args.shape[Dims-1], 3.5f);
    args.inv_stddev.resize(args.shape[Dims-1], 1.0/3.5f);
    return args;
  }
};


template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_PermuteOnly_ReversedDims {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    for (int d = 0; d < Dims; d++) {
      args.permuted_dims[d] = Dims-1-d;
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_PermuteAndSliceHalf_ReversedDims {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = input_shape[d]/4;
      args.shape[d] = args.padded_shape[d] = input_shape[d]/2;
      args.permuted_dims[d] = Dims-1-d;
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_PermuteAndSliceHalf_PermuteHW {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = input_shape[d]/4;
      args.shape[d] = args.padded_shape[d] = input_shape[d]/2;
      switch (d) {
        case 0:
          args.permuted_dims[d] = 1;
          break;
        case 1:
          args.permuted_dims[d] = 0;
          break;
        default:
          args.permuted_dims[d] = d;
          break;
      }
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_SliceFlipNormalizePermute_PermuteHWC2CHW {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = d == 0 || d == 1 ?
        input_shape[d]/2 : 0;
      args.shape[d] = args.padded_shape[d] = d == 0 || d == 1 ?
        input_shape[d]/2 : input_shape[d];
      args.flip[d] = d == 0 || d == 1;
      switch (d) {
        case 0:
          args.permuted_dims[d] = 2;
          break;
        case 1:
          args.permuted_dims[d] = 0;
          break;
        case 2:
          args.permuted_dims[d] = 1;
          break;
        default:
          args.permuted_dims[d] = d;
          break;
      }
    }
    args.mean.resize(args.shape[Dims-1], 50.0f);
    args.inv_stddev.resize(args.shape[Dims-1], 1.0/100.0f);
    return args;
  }
};

template <typename OutputType, int Dims, int PaddedDim, int PadSize>
struct SliceFlipNormPermArgsGen_OnlyPad_GivenDim {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape);
    args.padded_shape[PaddedDim] += PadSize;
    return args;
  }
};

using SLICE_FLIP_NORMALIZE_PERMUTE_TEST_TYPES = ::testing::Types<
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_CopyOnly<float, 3>>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_SliceOnly<float, 3>>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_SliceOnly_WithAnchor<float, 3>>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_FlipHW<float, 3>>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_FlipDim<float, 3, 0>>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_FlipDim<float, 3, 1>>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_FlipDim<float, 3, 2>>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_NormalizeOnly<float, 3>>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_NormalizeOnly_Scalar<float, 3>>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_NormalizeAndFlipDim<float, 3, 0>>,
    SliceTestArgs<int, float, 2, 1, 10,
      SliceFlipNormPermArgsGen_PermuteOnly_ReversedDims<float, 2>>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_PermuteOnly_ReversedDims<float, 3>>,
    SliceTestArgs<int, float, 2, 1, 1,
      SliceFlipNormPermArgsGen_PermuteOnly_ReversedDims<float, 2>, 10, 2>,
    SliceTestArgs<int, float, 2, 1, 1,
      SliceFlipNormPermArgsGen_PermuteOnly_ReversedDims<float, 2>, 2, 10>,
    SliceTestArgs<int, float, 2, 1, 1,
      SliceFlipNormPermArgsGen_PermuteAndSliceHalf_ReversedDims<float, 2>, 2, 10>,
    SliceTestArgs<int, float, 2, 1, 1,
      SliceFlipNormPermArgsGen_PermuteAndSliceHalf_PermuteHW<float, 2>, 2, 10>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_SliceFlipNormalizePermute_PermuteHWC2CHW<float, 3>, 2, 2>,
    SliceTestArgs<int, uint8_t, 3, 1, 3,
      SliceFlipNormPermArgsGen_PermuteAndSliceHalf_PermuteHW<uint8_t, 3>, 1000, 1000>,
    SliceTestArgs<uint8_t, float, 3, 1, 3,
      SliceFlipNormPermArgsGen_PermuteAndSliceHalf_PermuteHW<float, 3>, 1000, 1000>,
    SliceTestArgs<uint8_t, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_OnlyPad_GivenDim<float, 3, 2, 1>, 2, 2>,
    SliceTestArgs<uint8_t, float, 3, 1, 3,
      SliceFlipNormPermArgsGen_OnlyPad_GivenDim<float, 3, 2, 1>, 10, 10>,
    SliceTestArgs<uint8_t, float, 3, 1, 3,
      SliceFlipNormPermArgsGen_OnlyPad_GivenDim<float, 3, 1, 10>, 10, 10>,
    SliceTestArgs<uint8_t, float, 3, 1, 3,
      SliceFlipNormPermArgsGen_OnlyPad_GivenDim<float, 3, 0, 10>, 10, 10>,
    SliceTestArgs<int, bool, 3, 1, 2,
      SliceFlipNormPermArgsGen_SliceOnly<bool, 3>>
>;

using SLICE_FLIP_NORMALIZE_PERMUTE_TEST_TYPES_CPU_ONLY = ::testing::Types<
    SliceTestArgs<uint8_t, float16, 3, 1, 2,
      SliceFlipNormPermArgsGen_SliceOnly<float16, 3>>,
    SliceTestArgs<float16, uint8_t, 3, 1, 2,
      SliceFlipNormPermArgsGen_SliceOnly<uint8_t, 3>>
>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_KERNEL_TEST_H_
