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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CPU_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CPU_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/util/half.hpp"

namespace dali {
namespace kernels {

namespace detail {

template <bool NeedNormalize, typename OutputType, typename InputType>
inline void Fill(OutputType &destination, InputType element,
                 const float *mean, const float *inv_stddev) {
  if (NeedNormalize) {
    float fpout = (static_cast<float>(element) - (*mean)) * (*inv_stddev);
    destination = ConvertSat<OutputType>(fpout);
  } else {
    (void) mean;
    (void) inv_stddev;
    destination = ConvertSat<OutputType>(element);
  }
}

template <bool NeedNormalize, bool NeedPad, bool HasChannels, bool OutOfBounds,
          typename OutputType, typename InputType>
void SliceFlipNormalizePermutePadKernelImpl(OutputType *output,
                                            const InputType *input,
                                            const int64_t* in_strides,
                                            const int64_t* out_strides,
                                            const int64_t* anchor,
                                            const int64_t* in_shape,
                                            const int64_t* out_shape,
                                            const OutputType *fill_values,
                                            const float *mean,
                                            const float *inv_stddev,
                                            int channel_dim,  // negative if no channel dim or already processed
                                            std::integral_constant<int, 1>) {
  constexpr int d = 0;
  if (OutOfBounds) {
    for (int i = 0; i < out_shape[d]; i++) {
      output[i] = *fill_values;
      if (HasChannels && d == channel_dim) {
        fill_values++;
        mean++;
        inv_stddev++;
      }
    }
  } else {
    int in_idx = anchor[d];
    int out_idx = 0;

    if (NeedPad) {
      // out of bounds (left side)
      for (; in_idx < 0 && out_idx < out_shape[d]; in_idx++, out_idx++) {
        *output = *fill_values;
        output += out_strides[d];
        input  += in_strides[d];
        if (HasChannels && d == channel_dim) {
          fill_values++;
          mean++;
          inv_stddev++;
        }
      }
    }

    // within input bounds
    for (; in_idx < in_shape[d] && out_idx < out_shape[d]; in_idx++, out_idx++) {
      Fill<NeedNormalize>(*output, *input, mean, inv_stddev);
      output += out_strides[d];
      input  += in_strides[d];
      if (HasChannels && d == channel_dim) {
          fill_values++;
          mean++;
          inv_stddev++;
      }
    }

    if (NeedPad) {
      // out of bounds (right side)
      for (; out_idx < out_shape[d]; in_idx++, out_idx++) {
        *output = *fill_values;
        output += out_strides[d];
        input += in_strides[d];
        if (HasChannels && d == channel_dim) {
          fill_values++;
          mean++;
          inv_stddev++;
        }
      }
    }
  }
}

template <bool NeedNormalize, bool NeedPad, bool HasChannels, bool OutOfBounds,
          typename OutputType, typename InputType, int DimsLeft>
void SliceFlipNormalizePermutePadKernelImpl(OutputType *output,
                                            const InputType *input,
                                            const int64_t* in_strides,
                                            const int64_t* out_strides,
                                            const int64_t* anchor,
                                            const int64_t* in_shape,
                                            const int64_t* out_shape,
                                            const OutputType *fill_values,
                                            const float *mean,
                                            const float *inv_stddev,
                                            int channel_dim,  // negative if no channel dim or already processed
                                            std::integral_constant<int, DimsLeft>) {
  constexpr int d = 0;
  int in_idx = anchor[d];
  int out_idx = 0;

  if (NeedPad) {
    // out of bounds (left side)
    for (; in_idx < 0 && out_idx < out_shape[d]; in_idx++, out_idx++) {
      constexpr bool NewOutOfBounds = true;
      SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, NeedPad, HasChannels, NewOutOfBounds>(
          output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
          fill_values, mean, inv_stddev, channel_dim - 1, std::integral_constant<int, DimsLeft - 1>());
      output += out_strides[d];
      input  += in_strides[d];
      if (HasChannels && d == channel_dim) {
        fill_values++;
        mean++;
        inv_stddev++;
      }
    }
  }

  // within input bounds
  for (; in_idx < in_shape[d] && out_idx < out_shape[d]; in_idx++, out_idx++) {
    SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, NeedPad, HasChannels, OutOfBounds>(
        output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
        fill_values, mean, inv_stddev, channel_dim - 1, std::integral_constant<int, DimsLeft - 1>());
    output += out_strides[d];
    input  += in_strides[d];
    if (HasChannels && d == channel_dim) {
      fill_values++;
      mean++;
      inv_stddev++;
    }
  }

  if (NeedPad) {
    // out of bounds (right side)
    for (; out_idx < out_shape[d]; in_idx++, out_idx++) {
      constexpr bool NewOutOfBounds = true;
      SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, NeedPad, HasChannels, NewOutOfBounds>(
          output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
          fill_values, mean, inv_stddev, channel_dim - 1, std::integral_constant<int, DimsLeft - 1>());
      output += out_strides[d];
      input  += in_strides[d];
      if (HasChannels && d == channel_dim) {
        fill_values++;
        mean++;
        inv_stddev++;
      }
    }
  }
}

}  // namespace detail


template <int Dims, typename OutputType, typename InputType>
void SliceFlipNormalizePermutePadKernel(OutputType *output,
                                        const InputType *input,
                                        const TensorShape<Dims> &in_strides,
                                        const TensorShape<Dims> &out_strides,
                                        const TensorShape<Dims> &anchor,
                                        const TensorShape<Dims> &in_shape,
                                        const TensorShape<Dims> &out_shape,
                                        const OutputType *fill_values = nullptr,
                                        const float *mean = nullptr,
                                        const float *inv_stddev = nullptr,
                                        int channel_dim = -1) {  // negative if no channel dim or already processed
  bool need_pad = NeedPad(Dims, anchor.data(), in_shape.data(), out_shape.data());
  bool has_channels = channel_dim >= 0;
  bool need_normalize = (mean != nullptr && inv_stddev != nullptr);
  VALUE_SWITCH(need_normalize ? 1 : 0, NeedNormalizeInt, (0, 1), (
    VALUE_SWITCH(need_pad ? 1 : 0, NeedPadInt, (0, 1), (
      VALUE_SWITCH(has_channels ? 1 : 0, HasChannelsInt, (0, 1), (
        constexpr bool NeedNormalize = static_cast<bool>(NeedNormalizeInt);
        constexpr bool NeedPad = static_cast<bool>(NeedPadInt);
        constexpr bool HasChannels = static_cast<bool>(HasChannelsInt);
        constexpr bool OutOfBounds = false;
        detail::SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, NeedPad, HasChannels, OutOfBounds>(
            output, input, in_strides.data(), out_strides.data(), anchor.data(), in_shape.data(),
            out_shape.data(), fill_values, mean, inv_stddev, channel_dim, std::integral_constant<int, Dims>());
      ), ());
    ), ());
  ), ());
}

template <typename OutputType, typename InputType, int Dims>
class SliceFlipNormalizePermutePadCpu {
 public:
  using Args = SliceFlipNormalizePermutePadArgs<Dims>;

  KernelRequirements Setup(KernelContext &context,
                           const InTensorCPU<InputType, Dims> &in,
                           const Args &args) {
    KernelRequirements req;
    TensorShape<Dims> out_shape(args.shape);
    CheckValidOutputShape(in.shape, out_shape, args);
    out_shape = permute(out_shape, args.permuted_dims);
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, out_shape));
    return req;
  }

  void Run(KernelContext &context,
           OutTensorCPU<OutputType, Dims> &out,
           const InTensorCPU<InputType, Dims> &in,
           const Args &orig_args) {
    auto args = detail::ProcessArgs(orig_args, in.shape);

    int nvalues = args.fill_values.size();
    SmallVector<OutputType, 4> fill_values;  // TODO(janton): fix
    assert(!args.fill_values.empty());
    for (auto value : args.fill_values)
      fill_values.push_back(static_cast<OutputType>(value));

    int64_t in_size = volume(args.in_shape);
    int64_t out_size = volume(args.out_shape);
/*
    std::cout << "in_data:";
    for (int i = 0; i < in_size; i++)
      std::cout << " " << (int) in.data[i];
    std::cout << "\n";
*/
    SliceFlipNormalizePermutePadKernel(out.data, in.data + args.input_offset, args.in_strides, args.out_strides,
                                       args.anchor, args.in_shape, args.out_shape,
                                       fill_values.data(), args.mean.data(), args.inv_stddev.data(),
                                       args.channel_dim);
/*
    std::cout << "out_data:";
    for (int i = 0; i < out_size; i++)
      std::cout << " " << (int) out.data[i];
    std::cout << "\n";
*/
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CPU_H_
