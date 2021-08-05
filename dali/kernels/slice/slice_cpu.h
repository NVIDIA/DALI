// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <tuple>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/exec/engine.h"
#include "dali/kernels/common/split_shape.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_kernel_utils.h"


namespace dali {
namespace kernels {

static constexpr int kSliceCost = 2;  // compared to memcpy (heuristic)

namespace detail {

/**
 * @brief Optimized special case for the last two dimensions whith channel-last configuration
 */
template <typename OutputType, typename InputType, bool OutOfBounds, bool NeedPad>
void SliceKernelImplChannelLast(OutputType *output,
                                const InputType *input,
                                const int64_t* out_strides,
                                const int64_t* in_strides,
                                const int64_t* out_shape,
                                const int64_t* in_shape,
                                const int64_t* anchor,
                                const OutputType *fill_values,
                                int channel_dim,  // negative if no channel dim or already processed
                                std::integral_constant<bool, OutOfBounds>,
                                std::integral_constant<bool, NeedPad>) {
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
          output[out_c] = input[anchor_channel_in + in_c];

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
        out_row[j] = clamp<OutputType>(in_row[anchor[d + 1] + j]);
      }
    }
  }
}

template <typename OutputType, typename InputType, bool OutOfBounds, bool NeedPad>
void SliceKernelImpl(OutputType *output,
                     const InputType *input,
                     const int64_t* out_strides,
                     const int64_t* in_strides,
                     const int64_t* out_shape,
                     const int64_t* in_shape,
                     const int64_t* anchor,
                     const OutputType *fill_values,
                     int channel_dim,  // negative if no channel dim or already processed
                     std::integral_constant<int, 1>,
                     std::integral_constant<bool, OutOfBounds>,
                     std::integral_constant<bool, NeedPad>) {
  constexpr int d = 0;
  if (OutOfBounds) {
    for (int i = 0; i < out_shape[d]; i++) {
      output[i] = *fill_values;
      if (d == channel_dim)
        fill_values++;
    }
  } else {
    int in_idx = anchor[d];
    int out_idx = 0;

    if (NeedPad) {
      // out of bounds (left side)
      for (; in_idx < 0 && out_idx < out_shape[d]; in_idx++, out_idx++) {
        output[out_idx] = *fill_values;
        if (d == channel_dim)
          fill_values++;
      }
    }

    // within input bounds
    for (; in_idx < in_shape[d] && out_idx < out_shape[d]; in_idx++, out_idx++) {
      output[out_idx] = clamp<OutputType>(input[in_idx]);
      if (NeedPad && d == channel_dim)
        fill_values++;
    }

    if (NeedPad) {
      // out of bounds (right side)
      for (; out_idx < out_shape[d]; in_idx++, out_idx++) {
        output[out_idx] = *fill_values;
        if (d == channel_dim)
          fill_values++;
      }
    }
  }
}

template <typename OutputType, typename InputType, bool OutOfBounds, bool NeedPad, int DimsLeft>
void SliceKernelImpl(OutputType *output,
                     const InputType *input,
                     const int64_t* out_strides,
                     const int64_t* in_strides,
                     const int64_t* out_shape,
                     const int64_t* in_shape,
                     const int64_t* anchor,
                     const OutputType *fill_values,
                     int channel_dim,  // negative if no channel dim or already processed
                     std::integral_constant<int, DimsLeft>,
                     std::integral_constant<bool, OutOfBounds>,
                     std::integral_constant<bool, NeedPad>) {
  // Special case for last 2 dimensions with channel-last configuration
  if (DimsLeft == 2 && channel_dim == 1) {
    SliceKernelImplChannelLast(output, input, out_strides, in_strides, out_shape, in_shape, anchor,
                               fill_values, channel_dim,
                               std::integral_constant<bool, OutOfBounds>(),
                               std::integral_constant<bool, NeedPad>());
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
      SliceKernelImpl(output, input, out_strides + 1, in_strides + 1, out_shape + 1,
                      in_shape + 1, anchor + 1, fill_values, channel_dim - 1,
                      std::integral_constant<int, DimsLeft - 1>(),
                      std::integral_constant<bool, true>(),
                      std::integral_constant<bool, NeedPad>());
      output += out_strides[d];
      if (d == channel_dim)
        fill_values++;
    }
  }

  // within input bounds
  for (; in_idx < in_shape[d] && out_idx < out_shape[d]; in_idx++, out_idx++) {
    SliceKernelImpl(output, input, out_strides + 1, in_strides + 1, out_shape + 1,
                    in_shape + 1,  anchor + 1, fill_values, channel_dim - 1,
                    std::integral_constant<int, DimsLeft - 1>(),
                    std::integral_constant<bool, OutOfBounds>(),
                    std::integral_constant<bool, NeedPad>());
    output += out_strides[d];
    if (!OutOfBounds)
      input += in_strides[d];
    if (NeedPad && d == channel_dim)
      fill_values++;
  }

  if (NeedPad) {
    // out of bounds (right side)
    for (; out_idx < out_shape[d]; in_idx++, out_idx++) {
      SliceKernelImpl(output, input, out_strides + 1, in_strides + 1, out_shape + 1,
                      in_shape + 1, anchor + 1, fill_values, channel_dim - 1,
                      std::integral_constant<int, DimsLeft - 1>(),
                      std::integral_constant<bool, true>(),
                      std::integral_constant<bool, NeedPad>());
      output += out_strides[d];
      if (d == channel_dim)
        fill_values++;
    }
  }
}

}  // namespace detail


template <typename OutputType, typename InputType, int Dims>
void SliceKernel(OutputType *output,
                 const InputType *input,
                 const TensorShape<Dims> &out_strides,
                 const TensorShape<Dims> &in_strides,
                 const TensorShape<Dims> &out_shape,
                 const TensorShape<Dims> &in_shape,
                 const TensorShape<Dims> &anchor,
                 const OutputType *fill_values,
                 int channel_dim = -1) {  // negative if no channel dim or already processed
  bool need_pad = NeedPad(Dims, anchor.data(), in_shape.data(), out_shape.data());
  if (need_pad) {
    detail::SliceKernelImpl(
        output, input, out_strides.data(), in_strides.data(), out_shape.data(),
        in_shape.data(), anchor.data(), fill_values, channel_dim,
        std::integral_constant<int, Dims>(),
        std::integral_constant<bool, false>(),
        std::integral_constant<bool, true>());
  } else {
    detail::SliceKernelImpl(
        output, input, out_strides.data(), in_strides.data(), out_shape.data(),
        in_shape.data(), anchor.data(), fill_values, channel_dim,
        std::integral_constant<int, Dims>(),
        std::integral_constant<bool, false>(),
        std::integral_constant<bool, false>());
  }
}


/**
 * @brief Implementation of slice kernel with an execution engine.
 */
template <typename ExecutionEngine, typename OutputType, typename InputType, int Dims>
DLL_LOCAL  // workaround for GCC bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
void SliceKernel(ExecutionEngine &exec_engine,
                 OutputType* out_data,
                 const InputType *in_data,
                 const TensorShape<Dims> &out_strides,
                 const TensorShape<Dims> &in_strides,
                 const TensorShape<Dims> &out_shape,
                 const TensorShape<Dims> &in_shape,
                 const SliceArgs<OutputType, Dims> &args,
                 int min_blk_sz = kSliceMinBlockSize,
                 int req_nblocks = -1) {
  if (req_nblocks < 0)
    req_nblocks = exec_engine.NumThreads() * 8;

  // If the output and input data type is the same and the slice arguments take the whole extent
  // of the input, then we can simply run memcpy.
  if (CanRunPlainCopy<OutputType, InputType, Dims>(out_strides, in_strides,
                                                   out_shape, in_shape, args)) {
    int64_t total_sz = volume(out_shape);
    int64_t nbytes = total_sz * sizeof(OutputType);
    int nblocks = std::min<int64_t>(req_nblocks, div_ceil(total_sz, min_blk_sz));
    assert(nblocks > 0);

    int64_t prev_b_end = 0;
    for (int b = 0; b < nblocks; b++) {
      int64_t b_start = prev_b_end;
      int64_t b_end = prev_b_end = total_sz * (b + 1) / nblocks;
      int64_t b_nbytes = (b_end - b_start) * sizeof(OutputType);
      exec_engine.AddWork([=](int tid) {
        std::memcpy(out_data + b_start, in_data + b_start, b_nbytes);
      }, b_nbytes, false);  // do not start work immediately
    }
    return;
  }

  std::array<int, Dims> split_factor;
  uint64_t skip_dim_mask = args.channel_dim >= 0 ? (1_u64 << args.channel_dim) : 0;
  int nblocks = split_shape(split_factor, out_shape, req_nblocks, min_blk_sz, skip_dim_mask);

  if (nblocks == 1) {
    exec_engine.AddWork([=](int) {
      SliceKernel(out_data, in_data, out_strides, in_strides, out_shape, in_shape,
                  args.anchor, GetPtr<OutputType>(args.fill_values), args.channel_dim);
    }, kSliceCost * volume(out_shape), false);  // do not start work immediately
    return;
  }

  TensorShape<Dims> start;  // zero-filled
  const auto& end = out_shape;
  ForEachBlock(
    start, end, split_factor, 0, LastSplitDim(split_factor),
    [&](const TensorShape<Dims> &blk_start, const TensorShape<Dims> &blk_end) {
      auto output_ptr = out_data;
      TensorShape<Dims> blk_anchor;
      TensorShape<Dims> blk_shape;
      for (int d = 0; d < Dims; d++) {
        output_ptr += blk_start[d] * out_strides[d];
        blk_shape[d] = blk_end[d] - blk_start[d];
        blk_anchor[d] = args.anchor[d] + blk_start[d];
      }
      exec_engine.AddWork([=](int) {
        SliceKernel(output_ptr, in_data, out_strides, in_strides, blk_shape, in_shape,
                    blk_anchor, GetPtr<OutputType>(args.fill_values), args.channel_dim);
      }, kSliceCost * volume(blk_shape), false);  // do not start work immediately
    });
  // scheduled work does not start until user calls Run()
}

/**
 * @brief Specialization for SequentialExecutionEngine.
 *        The slice is processed without partitioning.
 */
template <typename OutputType, typename InputType, int Dims>
void SliceKernel(SequentialExecutionEngine &exec_engine,
                 OutputType* out_data,
                 const InputType *in_data,
                 const TensorShape<Dims> &out_strides,
                 const TensorShape<Dims> &in_strides,
                 const TensorShape<Dims> &out_shape,
                 const TensorShape<Dims> &in_shape,
                 const SliceArgs<OutputType, Dims> &args,
                 int /* min_blk_sz */ = -1, int /* req_nblocks */ = -1) {
  (void) exec_engine;

  // If the output and input data type is the same and the slice arguments take the whole extent
  // of the input, then we can simply run memcpy.
  if (CanRunPlainCopy<OutputType, InputType, Dims>(out_strides, in_strides,
                                                   out_shape, in_shape, args)) {
    std::memcpy(out_data, in_data, sizeof(OutputType) * volume(out_shape));
    return;
  }

  SliceKernel(out_data, in_data, out_strides, in_strides, out_shape, in_shape,
              args.anchor, GetPtr<OutputType>(args.fill_values), args.channel_dim);
}

template <typename OutputType, typename InputType, int Dims>
class SliceCPU {
 public:
  static_assert(Dims >= 0, "Dims must be >= 0");

  KernelRequirements Setup(KernelContext &context,
                           const InTensorCPU<InputType, Dims> &in,
                           const SliceArgs<OutputType, Dims> &slice_args) {
    KernelRequirements req;
    auto shape = GetOutputShape(in.shape, slice_args);
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, shape));
    return req;
  }

  /**
   * @brief Schedules the kernel work with an execution engine.
   *
   *        The work is only schedule and does not start until the user calls RunAll()
   *        on the execution engine.
   *
   *        The user is responsible to synchronize with the execution engine.
   *
   *        For execution engines other than SequentialExecutionEngine, the algorithm will try
   *        to split the slice into similar sized blocks until we either reach a minimum of ``req_nblocks``
   *        or the block volume is smaller than the minimum practical size, ``min_blk_sz``.
   * @param context Kernel context
   * @param out Output tensor view
   * @param in Input tensor view
   * @param slice_args Slice arguments
   * @param min_blk_sz Minimum practical block size
   * @param req_nblocks Requested number of blocks. By default the requested number of blocks
   *                    is calculated as ``num_threads * 8``
   */
  template <typename ExecutionEngine>
  void Schedule(KernelContext &context,
                OutTensorCPU<OutputType, Dims> out,
                InTensorCPU<InputType, Dims> in,
                const SliceArgs<OutputType, Dims> &args,
                ExecutionEngine &exec_engine,
                int min_blk_sz = kSliceMinBlockSize, int req_nblocks = -1) {
    auto out_strides = GetStrides(out.shape);
    auto in_strides = GetStrides(in.shape);

    // fill values should not be empty. It should be left default if not used
    assert(!args.fill_values.empty());
    const OutputType* fill_values = args.fill_values.data();
    int fill_values_size = args.fill_values.size();
    if (fill_values_size > 1) {
      DALI_ENFORCE(args.channel_dim >= 0 && args.channel_dim < Dims,
        "Channels dimension needs to be specified if multi-channel fill_values is provided");
      DALI_ENFORCE(fill_values_size == out.shape[args.channel_dim],
        "Multi-channel fill value does not match the number of channels in the input");
    }

    SliceKernel(exec_engine, out.data, in.data, out_strides, in_strides, out.shape, in.shape, args,
                min_blk_sz, req_nblocks);
  }

  /**
   * @brief Run the kernel
   */
  void Run(KernelContext &context,
           OutTensorCPU<OutputType, Dims> out,
           InTensorCPU<InputType, Dims> in,
           const SliceArgs<OutputType, Dims> &slice_args) {
    SequentialExecutionEngine engine;
    Schedule(context, out, in, slice_args, engine);  // work is run synchronously, no need to wait
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_CPU_H_
