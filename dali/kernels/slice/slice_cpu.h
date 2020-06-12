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

#ifndef DALI_KERNELS_SLICE_SLICE_CPU_H_
#define DALI_KERNELS_SLICE_SLICE_CPU_H_

#include <utility>
#include <tuple>
#include <vector>
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {

namespace detail {

struct ClampPolicy {
  template <typename OutputType, typename InputType>
  static inline void Fill(OutputType &destination, InputType element,
                          const float *mean, const float *inv_stddev) {
    (void) mean;
    (void) inv_stddev;
    if (std::is_integral<OutputType>::value && std::is_floating_point<InputType>::value) {
      destination = clamp<OutputType>(std::roundf(element));
    } else {
      destination = clamp<OutputType>(element);
    }
  }
};

struct NormalizePolicy {
  template <typename OutputType, typename InputType>
  static inline void Fill(OutputType &destination, InputType element,
                          const float *mean, const float *inv_stddev) {
    float fpout = (static_cast<float>(element) - (*mean)) * (*inv_stddev);
    if (std::is_integral<OutputType>::value) {
      destination = clamp<OutputType>(std::roundf(fpout));
    } else {
      destination = clamp<OutputType>(fpout);
    }
  }
};

/**
 * @brief Fills output with nchannel values repeatedly
 */
template <typename T>
void PadFill(T *output, const T *fill_values, int64_t npixels, int64_t nchannels) {
  int64_t n = npixels * nchannels;
  int64_t i = 0;
  for (; i < nchannels; i++)
    output[i] = fill_values[i];
  for (; i < n; i++)
    output[i] = output[i - nchannels];
}

inline std::tuple<int64_t, int64_t, int64_t> CalcPadCopyExtents(int64_t anchor,
                                                                int64_t in_extent,
                                                                int64_t out_extent) {
  int64_t pad_before = std::min(out_extent, std::max<int64_t>(0, -anchor));
  int64_t to_copy = std::max<int64_t>(
      0, std::min(in_extent - std::max<int64_t>(0, anchor), out_extent - pad_before));
  int64_t pad_after = out_extent - pad_before - to_copy;
  return std::tuple<int64_t, int64_t, int64_t>{pad_before, to_copy, pad_after};
}

/**
 * @brief Optimized special case for the last two dimensions whith channel-last configuration
 */
template <typename Policy, bool OutOfBounds, bool NeedPad, bool HasChannels, 
          typename OutputType, typename InputType>
void SliceKernelImplChannelLast(OutputType *output,
                                const InputType *input,
                                const int64_t* in_strides,
                                const int64_t* out_strides,
                                const int64_t* anchor,
                                const int64_t* in_shape,
                                const int64_t* out_shape,
                                const OutputType *fill_values,
                                const float *mean,
                                const float *inv_stddev,
                                int channel_dim) {  // negative if no channel dim or already processed
  constexpr int d = 0;
  assert(channel_dim == 1);
  int64_t out_nchannels = out_shape[channel_dim];
  int64_t in_nchannels = in_shape[channel_dim];
  int64_t npixels = out_shape[d];

  if (NeedPad) {
    // If the whole row is out of bounds, just fill
    if (OutOfBounds) {
      PadFill(output, fill_values, npixels, out_nchannels);
      return;
    }

    // Calculate number of pixels to pad on the left and right, and the number of pixels to be
    // copied
    int64_t pad_pixels_before, copy_pixels, pad_pixels_after;
    std::tie(pad_pixels_before, copy_pixels, pad_pixels_after) =
        CalcPadCopyExtents(anchor[d], in_shape[d], out_shape[d]);

    // Padding pixels on the left, if needed
    if (pad_pixels_before > 0) {
      PadFill(output, fill_values, pad_pixels_before, out_nchannels);
      output += pad_pixels_before * out_strides[d];
    }

    // If the anchor is positive, advance the input pointer
    if (anchor[d] > 0)
      input += anchor[d] * in_strides[d];

    bool channel_dim_unchanged = in_nchannels == out_nchannels && anchor[channel_dim] == 0;
    if (channel_dim_unchanged) {
      auto n = copy_pixels * out_nchannels;
      for (int64_t i = 0; i < n; i++)
        output[i] = input[i];
      output += n;
    } else {
      // Calculate number of channels to pad on the left, right and the number of channels to be
      // copied
      int64_t pad_channels_before, copy_channels, pad_channels_after;
      std::tie(pad_channels_before, copy_channels, pad_channels_after) =
          CalcPadCopyExtents(anchor[channel_dim], in_nchannels, out_nchannels);
      int64_t anchor_channel_in = std::max<int64_t>(0, anchor[channel_dim]);
      // Copy pixels with potential padding on the channel dimension
      for (int64_t i = 0; i < copy_pixels; i++) {
        int64_t out_c = 0;
        for (; out_c < pad_channels_before; out_c++)
          output[out_c] = fill_values[out_c];

        for (int64_t in_c = 0; in_c < copy_channels; in_c++, out_c++)
          Policy::Fill(output[out_c], input[anchor_channel_in + in_c], mean, inv_stddev);

        for (; out_c < out_nchannels; out_c++)
          output[out_c] = fill_values[out_c];

        output += out_nchannels;
        input += in_nchannels;
      }
    }

    // Padding pixels on the right, if needed
    if (pad_pixels_after > 0) {
      PadFill(output, fill_values, pad_pixels_after, out_nchannels);
      output += pad_pixels_after * out_strides[d];
    }
  } else {  // NeedPad = false
    assert(out_strides[d + 1] == 1);
    assert(in_strides[d + 1] == 1);
    for (int64_t i = 0; i < out_shape[d]; i++) {
      auto *out_row = output + i * out_strides[d];
      auto *in_row = input + (anchor[d] + i) * in_strides[d];
      for (int64_t j = 0; j < out_shape[d + 1]; j++) {
        Policy::Fill(out_row[j], in_row[anchor[d + 1] + j], mean, inv_stddev);
      }
    }
  }
}

template <typename Policy, bool OutOfBounds, bool NeedPad, bool HasChannels,
          typename OutputType, typename InputType>
void SliceKernelImpl(OutputType *output,
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
        output[out_idx] = *fill_values;
        if (HasChannels && d == channel_dim) {
          fill_values++;
          mean++;
          inv_stddev++;
        }
      }
    }

    // within input bounds
    for (; in_idx < in_shape[d] && out_idx < out_shape[d]; in_idx++, out_idx++) {
      Policy::Fill(output[out_idx], input[in_idx], mean, inv_stddev);
      if (HasChannels && d == channel_dim) {
          fill_values++;
          mean++;
          inv_stddev++;
      }
    }

    if (NeedPad) {
      // out of bounds (right side)
      for (; out_idx < out_shape[d]; in_idx++, out_idx++) {
        output[out_idx] = *fill_values;
        if (HasChannels && d == channel_dim) {
          fill_values++;
          mean++;
          inv_stddev++;
        }
      }
    }
  }
}

template <typename Policy, bool OutOfBounds, bool NeedPad, bool HasChannels,
          typename OutputType, typename InputType, int DimsLeft>
void SliceKernelImpl(OutputType *output,
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
  // Special case for last 2 dimensions with channel-last configuration
  if (DimsLeft == 2 && channel_dim == 1) {
    SliceKernelImplChannelLast<Policy, OutOfBounds, NeedPad, HasChannels>(
        output, input, in_strides, out_strides, anchor, in_shape, out_shape, fill_values, mean,
        inv_stddev, channel_dim);
    return;
  }

  constexpr int d = 0;
  int in_idx = anchor[d];
  int out_idx = 0;

  if (anchor[d] > 0 && anchor[d] < in_shape[d])
    input += anchor[d] * in_strides[d];

  if (NeedPad) {
    // out of bounds (left side)
    for (; in_idx < 0 && out_idx < out_shape[d]; in_idx++, out_idx++) {
      SliceKernelImpl<Policy, true, NeedPad, HasChannels>(
          output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
          fill_values, mean, inv_stddev, channel_dim - 1, std::integral_constant<int, DimsLeft - 1>());
      output += out_strides[d];
      if (HasChannels && d == channel_dim) {
        fill_values++;
        mean++;
        inv_stddev++;
      }
    }
  }

  // within input bounds
  for (; in_idx < in_shape[d] && out_idx < out_shape[d]; in_idx++, out_idx++) {
    SliceKernelImpl<Policy, OutOfBounds, NeedPad, HasChannels>(
        output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
        fill_values, mean, inv_stddev, channel_dim - 1, std::integral_constant<int, DimsLeft - 1>());
    output += out_strides[d];
    if (!OutOfBounds)
      input += in_strides[d];
    if (HasChannels && d == channel_dim) {
      fill_values++;
      mean++;
      inv_stddev++;
    }
  }

  if (NeedPad) {
    // out of bounds (right side)
    for (; out_idx < out_shape[d]; in_idx++, out_idx++) {
      SliceKernelImpl<Policy, true, NeedPad, HasChannels>(
          output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
          fill_values, mean, inv_stddev, channel_dim - 1, std::integral_constant<int, DimsLeft - 1>());
      output += out_strides[d];
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
void SliceKernel(OutputType *output,
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
  VALUE_SWITCH(need_pad ? 1 : 0, NeedPadInt, (0, 1), (
    VALUE_SWITCH(has_channels ? 1 : 0, HasChannelsInt, (0, 1), (
      constexpr bool NeedPad = static_cast<bool>(NeedPadInt);
      constexpr bool HasChannels = static_cast<bool>(HasChannelsInt);
      if (need_normalize) {
        detail::SliceKernelImpl<detail::NormalizePolicy, false, NeedPad, HasChannels>(
            output, input, in_strides.data(), out_strides.data(), anchor.data(), in_shape.data(),
            out_shape.data(), fill_values, mean, inv_stddev, channel_dim, std::integral_constant<int, Dims>());
      } else {
        detail::SliceKernelImpl<detail::ClampPolicy, false, NeedPad, HasChannels>(
            output, input, in_strides.data(), out_strides.data(), anchor.data(), in_shape.data(),
            out_shape.data(), fill_values, mean, inv_stddev, channel_dim, std::integral_constant<int, Dims>());
      }
    ), ());
  ), ());
}

template <typename OutputType, typename InputType, int Dims>
class SliceCPU {
 public:
  KernelRequirements Setup(KernelContext &context,
                           const InTensorCPU<InputType, Dims> &in,
                           const SliceArgs<OutputType, Dims> &slice_args) {
    KernelRequirements req;
    auto shape = GetOutputShape(in.shape, slice_args);
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, shape));
    return req;
  }

  void Run(KernelContext &context,
           OutTensorCPU<OutputType, Dims> &out,
           const InTensorCPU<InputType, Dims> &in,
           const SliceArgs<OutputType, Dims> &slice_args) {
    const auto &in_shape = in.shape;
    const auto &out_shape = out.shape;
    const auto &anchor = slice_args.anchor;
    auto in_strides = GetStrides(in_shape);
    auto out_strides = GetStrides(out_shape);
    const InputType *in_ptr = in.data;
    OutputType *out_ptr = out.data;

    // fill values should not be empty. It should be left default if not used
    assert(!slice_args.fill_values.empty());
    int channel_dim = -1;  // channel dim is only used if a multi-channel fill_values is provided
    const OutputType* fill_values = slice_args.fill_values.data();
    int fill_values_size = slice_args.fill_values.size();
    
    const float *mean = nullptr;
    const float *inv_stddev = nullptr;
    DALI_ENFORCE(slice_args.mean.size() == slice_args.inv_stddev.size());
    int norm_args_size = slice_args.mean.size();
    if (norm_args_size > 0) {
      mean = slice_args.mean.data();
      inv_stddev = slice_args.inv_stddev.data();  
    }

    if (fill_values_size > 1 || norm_args_size > 1) {
      channel_dim = slice_args.channel_dim;
      DALI_ENFORCE(channel_dim >= 0 && channel_dim < Dims,
        "Channels dimension needs to be specified if multi-channel fill_values is provided");
    }
    if (fill_values_size > 1)
      DALI_ENFORCE(fill_values_size == out_shape[channel_dim],
        "Multi-channel fill value does not match the number of channels in the input");

    if (norm_args_size > 0)
      DALI_ENFORCE(norm_args_size == out_shape[channel_dim],
        "Size of the normalization arguments does not match the number of channels in the input");

    SliceKernel<Dims>(out_ptr, in_ptr, in_strides, out_strides, anchor, in_shape,
                      out_shape, fill_values, mean, inv_stddev, channel_dim);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_CPU_H_
