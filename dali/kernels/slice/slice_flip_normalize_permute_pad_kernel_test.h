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
      auto out_shape = shape;
      for (int d = 0; d < Dims; d++) {
        auto perm_d = args[i].permuted_dims[d];
        out_shape[d] = shape[perm_d];
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

      auto fill_values = args[i].fill_values;
      auto channel_dim = args[i].channel_dim;

      // normalization
      auto mean = args[i].mean;
      auto inv_stddev = args[i].inv_stddev;
      ASSERT_EQ(mean.size(), inv_stddev.size());
      bool need_normalize = !mean.empty();
      if (mean.size() > 1 && fill_values.size() == 1) {
        for (size_t i = 1; i < mean.size(); i++)
          fill_values.push_back(fill_values[0]);
      }
      int per_ch_arg_size = std::max(mean.size(), fill_values.size());
      if (per_ch_arg_size > 1 && channel_dim == -1)
        channel_dim = Dims - 1;

      // Naive implementation just for test purposes
      int i_c = 0;
      int64_t total_size = volume(out_shape);
      for (int64_t out_idx = 0; out_idx < total_size; out_idx++) {
        int64_t idx = out_idx;
        int64_t in_idx = 0;
        bool out_of_bounds = false;
        for (int d = 0; d < Dims; d++) {
          int i_d = idx / out_strides[d];
          idx = idx % out_strides[d];
          auto perm_d = permuted_dims[d];
          if (perm_d == channel_dim) {
            i_c = i_d;
            ASSERT_EQ(in_shape[perm_d], per_ch_arg_size);
            ASSERT_TRUE(i_c >= 0 && i_c < per_ch_arg_size);
          }
          int in_i_d = flip[perm_d] ? anchor[perm_d] + slice_shape[perm_d] - 1 - i_d
                                    : anchor[perm_d] + i_d;
          out_of_bounds |= in_i_d < 0 || in_i_d >= in_shape[perm_d];
          if (!out_of_bounds)
            in_idx += in_i_d * in_strides[perm_d];
        }

        if (out_of_bounds) {
          out_tensor[out_idx] = fill_values[i_c];
        } else if (need_normalize) {
          float fpout = (static_cast<float>(in_tensor[in_idx]) - mean[i_c]) * inv_stddev[i_c];
          out_tensor[out_idx] = ConvertSat<OutputType>(fpout);
        } else {
          out_tensor[out_idx] = ConvertSat<OutputType>(in_tensor[in_idx]);
        }
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
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_SliceOnly {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    auto shape = input_shape;
    shape[0] /= 2;
    shape[1] /= 2;
    SliceFlipNormalizePermutePadArgs<Dims> args(shape, input_shape);
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_SliceOnly_WithAnchor {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    auto shape = input_shape;
    shape[0] = input_shape[0]/2;
    shape[1] = input_shape[0]/2;
    SliceFlipNormalizePermutePadArgs<Dims> args(shape, input_shape);
    args.anchor[0] = input_shape[0]/2;
    args.anchor[1] = input_shape[1]/2;
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_FlipHW {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    // assuming last dims are HWC, flip H and W
    args.flip[Dims-2] = true;
    args.flip[Dims-3] = true;
    return args;
  }
};

template <typename OutputType, int Dims, int FlipDim>
struct SliceFlipNormPermArgsGen_FlipDim {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    args.flip[FlipDim] = true;
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_NormalizeOnly {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    args.mean.resize(args.shape[Dims-1]);
    args.inv_stddev.resize(args.shape[Dims-1]);
    for (int i = 0; i < args.shape[Dims-1]; i++) {
      args.mean[i] = 3.5f + 0.1f * i;
      args.inv_stddev[i] = 1 / (8.0f + 0.1f * i);
    }
    args.channel_dim = Dims - 1;
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_NormalizeOnly_Scalar {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    args.mean = { 3.5f };
    args.inv_stddev = { 1.f / 8.0f };
    return args;
  }
};

template <typename OutputType, int Dims, int FlipDim>
struct SliceFlipNormPermArgsGen_NormalizeAndFlipDim {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    args.flip[FlipDim] = true;
    args.mean.resize(args.shape[Dims - 1], 3.5f);
    args.inv_stddev.resize(args.shape[Dims - 1], 1.0/3.5f);
    args.channel_dim = Dims - 1;
    return args;
  }
};


template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_PermuteOnly_ReversedDims {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    for (int d = 0; d < Dims; d++) {
      args.permuted_dims[d] = Dims-1-d;
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_PermuteAndSliceHalf_ReversedDims {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = input_shape[d]/4;
      args.shape[d] = input_shape[d]/2;
      args.permuted_dims[d] = Dims-1-d;
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_PermuteAndSliceHalf_PermuteHW {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = input_shape[d]/4;
      args.shape[d] = input_shape[d]/2;
      args.permuted_dims[0] = 1;
      args.permuted_dims[1] = 0;
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_SliceFlipNormalizePermute_PermuteHWC2CHW {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    int dim_map[] = { 2, 0, 1 };
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = d == 0 || d == 1 ?
        input_shape[d]/2 : 0;
      args.shape[d] = d == 0 || d == 1 ?
        input_shape[d]/2 : input_shape[d];
      args.flip[d] = d == 0 || d == 1;
      args.permuted_dims[d] = d < 3 ? dim_map[d] : d;
    }
    args.mean.resize(args.shape[Dims - 1], 50.0f);
    args.inv_stddev.resize(args.shape[Dims - 1], 1.0/100.0f);
    args.channel_dim = Dims - 1;
    return args;
  }
};

template <typename OutputType, int Dims, bool flip_0 = true, bool flip_1 = true>
struct SliceFlipNormPermArgsGen_SlicePadFlip {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    args.anchor[0] = input_shape[0]/2;
    args.flip[0] = true;
    args.anchor[1] = input_shape[1]/2;
    args.flip[1] = true;
    return args;
  }
};


template <typename OutputType, int Dims, bool MultiChannel = true>
struct SliceFlipNormPermArgsGen_SlicePadFlipNormalizePermute_PermuteHWC2CHW {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    int dim_map[] = { 2, 0, 1 };
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = d == 0 || d == 1 ? input_shape[d]/2 : 0;
      args.shape[d] = input_shape[d];
      args.flip[d] = d == 0 || d == 1;
      args.permuted_dims[d] = d < 3 ? dim_map[d] : d;
    }
    if (MultiChannel) {
      int perm_channel_dim = args.permuted_dims[Dims - 1];
      int nchannels = args.shape[perm_channel_dim];
      args.mean.resize(nchannels, 50.0f);
      args.inv_stddev.resize(nchannels, 1.0/100.0f);
      args.fill_values.resize(nchannels, 0.5f);
      args.channel_dim = Dims - 1;
    } else {
      args.mean = {50.0f};
      args.inv_stddev = {1.0/100.0f};
      args.fill_values = {0.5f};
      args.channel_dim = -1;
    }
    return args;
  }
};

template <typename OutputType, int Dims>
struct SliceFlipNormPermArgsGen_SlicePadPermuteAll {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    int dim_map[] = { 2, 0, 1 };
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = input_shape[d]/2;
      args.shape[d] = input_shape[d];
      args.permuted_dims[d] = d < 3 ? dim_map[d] : d;
    }
    args.fill_values = {0.5f};
    args.channel_dim = -1;
    return args;
  }
};


template <typename OutputType, int Dims, bool MultiChannel = true>
struct SliceFlipNormPermArgsGen_SlicePadNormalizePermute_PermuteHWC2CHW {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    int dim_map[] = { 2, 0, 1 };
    for (int d = 0; d < Dims; d++) {
      args.anchor[d] = d == 0 || d == 1 ? input_shape[d]/2 : 0;
      args.shape[d] = input_shape[d];
      args.flip[d] = false;
      args.permuted_dims[d] = d < 3 ? dim_map[d] : d;
    }
    if (MultiChannel) {
      int perm_channel_dim = args.permuted_dims[Dims - 1];
      int nchannels = args.shape[perm_channel_dim];
      args.mean.resize(nchannels, 50.0f);
      args.inv_stddev.resize(nchannels, 1.0/100.0f);
      args.fill_values.resize(nchannels, 0.5f);
      args.channel_dim = Dims - 1;
    } else {
      args.mean = {50.0f};
      args.inv_stddev = {1.0/100.0f};
      args.fill_values = {0.5f};
      args.channel_dim = -1;
    }
    return args;
  }
};

template <typename OutputType, int Dims, int PaddedDim, int PadSize>
struct SliceFlipNormPermArgsGen_OnlyPad_GivenDim {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    args.shape[PaddedDim] += PadSize;
    return args;
  }
};

template <typename OutputType, int Dims, int FlipDim>
struct SliceFlipNormPermArgsGen_FlipPad_GivenDim {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    args.flip[FlipDim] = true;
    return args;
  }
};

template <typename OutputType, int Dims = 3>
struct ArgsGen_SingleValuePad_PermuteHWC2CHW {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    args.anchor[0] = -input_shape[0] / 2;
    args.anchor[1] = -input_shape[1] / 2;
    args.anchor[2] = 0;
    args.shape[0] = 2 * input_shape[0];
    args.shape[1] = 2 * input_shape[1];
    args.shape[2] = input_shape[2];
    args.fill_values = {0.5f};
    args.channel_dim = -1;
    args.permuted_dims = {2, 0, 1};
    return args;
  }
};

template <typename OutputType, int Dims = 3>
struct ArgsGen_SliceNormalizeFlip_PadChannels {
  SliceFlipNormalizePermutePadArgs<Dims> Get(const TensorShape<Dims>& input_shape) {
    SliceFlipNormalizePermutePadArgs<Dims> args(input_shape, input_shape);
    args.anchor[0] = input_shape[0] / 4;
    args.anchor[1] = input_shape[1] / 4;
    args.anchor[2] = 0;
    args.shape[0] = input_shape[0]/2;
    args.shape[1] = input_shape[1]/2;
    args.shape[2] = input_shape[2] + 1;
    args.flip[0] = true;
    args.mean = {127.0f};
    args.inv_stddev = {1/127.0f};
    args.channel_dim = -1;
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
    SliceTestArgs<int, float, 2, 1, 2,
      SliceFlipNormPermArgsGen_SlicePadFlip<float, 2, false, true>, 2, 2>,
    SliceTestArgs<int, float, 2, 1, 2,
      SliceFlipNormPermArgsGen_SlicePadFlip<float, 2, true, false>, 2, 2>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_SlicePadFlip<float, 3, true, true>, 2, 2>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_SlicePadNormalizePermute_PermuteHWC2CHW<float, 3>, 2, 2>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_SlicePadNormalizePermute_PermuteHWC2CHW<float, 3, false>, 2, 2>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_SlicePadFlipNormalizePermute_PermuteHWC2CHW<float, 3>, 2, 2>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_SlicePadFlipNormalizePermute_PermuteHWC2CHW<float, 3, false>, 2, 2>,
    SliceTestArgs<int, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_SlicePadPermuteAll<float, 3>, 2, 2>,
    SliceTestArgs<int, uint8_t, 3, 1, 3,
      SliceFlipNormPermArgsGen_PermuteAndSliceHalf_PermuteHW<uint8_t, 3>, 1000, 1000>,
    SliceTestArgs<uint8_t, float, 3, 1, 3,
      SliceFlipNormPermArgsGen_PermuteAndSliceHalf_PermuteHW<float, 3>, 1000, 1000>,
    SliceTestArgs<uint8_t, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_FlipPad_GivenDim<float, 3, 0>, 2, 2>,
    SliceTestArgs<uint8_t, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_FlipPad_GivenDim<float, 3, 1>, 2, 2>,
    SliceTestArgs<uint8_t, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_FlipPad_GivenDim<float, 3, 2>, 2, 2>,
    SliceTestArgs<uint8_t, float, 3, 1, 2,
      SliceFlipNormPermArgsGen_OnlyPad_GivenDim<float, 3, 2, 1>, 2, 2>,
    SliceTestArgs<uint8_t, float, 3, 1, 3,
      SliceFlipNormPermArgsGen_OnlyPad_GivenDim<float, 3, 2, 1>, 10, 10>,
    SliceTestArgs<uint8_t, float, 3, 1, 3,
      SliceFlipNormPermArgsGen_OnlyPad_GivenDim<float, 3, 1, 10>, 10, 10>,
    SliceTestArgs<uint8_t, float, 3, 1, 3,
      SliceFlipNormPermArgsGen_OnlyPad_GivenDim<float, 3, 0, 10>, 10, 10>,
    SliceTestArgs<int, bool, 3, 1, 2,
      SliceFlipNormPermArgsGen_SliceOnly<bool, 3>>,
    SliceTestArgs<int, float, 3, 1, 10,
      ArgsGen_SingleValuePad_PermuteHWC2CHW<float, 3>, 10, 10>,
    SliceTestArgs<int, float, 3, 1, 10,
      ArgsGen_SliceNormalizeFlip_PadChannels<float, 3>, 10, 10>
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
