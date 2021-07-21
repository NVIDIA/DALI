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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CPU_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CPU_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/exec/engine.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/split_shape.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/util/half.hpp"

namespace dali {
namespace kernels {

static constexpr int kSliceFlipNormPermCost = 8;  // compared to memcpy (heuristic)

namespace detail {

template <bool NeedNormalize, typename OutputType, typename InputType>
inline void Fill(OutputType &destination, InputType element,
                 const float *mean, const float *inv_stddev) {
  if (NeedNormalize) {
    float fpout = (static_cast<float>(element) - (*mean)) * (*inv_stddev);
    destination = ConvertSat<OutputType>(fpout);
  } else {
    destination = ConvertSat<OutputType>(element);
  }
}

template <bool NeedNormalize, bool HasChannels, typename OutputType, typename InputType>
void SliceFlipNormalizePermuteKernelImpl(
    OutputType *output, const InputType *input, const int64_t *in_strides,
    const int64_t *out_strides, const int64_t *anchor, const int64_t *in_shape,
    const int64_t *out_shape, const float *mean, const float *inv_stddev,
    int channel_dim,  // negative if no channel dim or already processed
    std::integral_constant<int, 1>) {
  constexpr int d = 0;
  // Note: out_strides[d] is 1 so we can just do output++ in the loops
  if (HasChannels && d == channel_dim) {
    for (int64_t i = 0; i < out_shape[d]; i++, input += in_strides[d])
      Fill<NeedNormalize>(*output++, *input, mean++, inv_stddev++);
  } else {
    for (int64_t i = 0; i < out_shape[d]; i++, input += in_strides[d])
      Fill<NeedNormalize>(*output++, *input, mean, inv_stddev);
  }
}

template <bool NeedNormalize, bool HasChannels, typename OutputType,
          typename InputType, int DimsLeft>
void SliceFlipNormalizePermuteKernelImpl(
    OutputType *output, const InputType *input, const int64_t *in_strides,
    const int64_t *out_strides, const int64_t *anchor, const int64_t *in_shape,
    const int64_t *out_shape, const float *mean, const float *inv_stddev,
    int channel_dim,  // negative if no channel dim or already processed
    std::integral_constant<int, DimsLeft>) {
  constexpr int d = 0;
  if (HasChannels && d == channel_dim) {
    for (int64_t i = 0; i < out_shape[d]; i++, output += out_strides[d], input += in_strides[d])
      SliceFlipNormalizePermuteKernelImpl<NeedNormalize, HasChannels>(
          output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
          mean++, inv_stddev++, channel_dim - 1, std::integral_constant<int, DimsLeft - 1>());
  } else {
    for (int64_t i = 0; i < out_shape[d]; i++, output += out_strides[d], input += in_strides[d])
      SliceFlipNormalizePermuteKernelImpl<NeedNormalize, HasChannels>(
          output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
          mean, inv_stddev, channel_dim - 1, std::integral_constant<int, DimsLeft - 1>());
  }
}

template <bool NeedNormalize, bool HasChannels, bool OutOfBounds, typename OutputType,
          typename InputType>
void SliceFlipNormalizePermutePadKernelImpl(
    OutputType *output, const InputType *input, const int64_t *in_strides,
    const int64_t *out_strides, const int64_t *anchor, const int64_t *in_shape,
    const int64_t *out_shape, const OutputType *fill_values, const float *mean,
    const float *inv_stddev,
    int channel_dim,  // negative if no channel dim or already processed
    std::integral_constant<int, 1>) {
  constexpr int d = 0;
  // Note: out_strides[d] is 1 so we can just do output++ in the loops
  if (OutOfBounds) {
    if (HasChannels && d == channel_dim) {
      for (int i = 0; i < out_shape[d]; i++)
        output[i] = *fill_values++;
    } else {
      for (int i = 0; i < out_shape[d]; i++)
        output[i] = *fill_values;
    }
  } else  {
    int64_t pad_before, slice, pad_after;
    std::tie(pad_before, slice, pad_after) =
      CalcPadCopyExtents(anchor[d], in_shape[d], out_shape[d]);
    if (in_strides[d] < 0) std::swap(pad_before, pad_after);

    // out of bounds (left side)
    if (pad_before > 0) {
      if (HasChannels && d == channel_dim) {
        for (int64_t i = 0; i < pad_before; i++) {
          *output++ = *fill_values++;
        }
        mean       += pad_before;
        inv_stddev += pad_before;
      } else {
        for (int64_t i = 0; i < pad_before; i++)
          *output++ = *fill_values;
      }
      input += pad_before * in_strides[d];
    }

    // within input bounds
    if (HasChannels && d == channel_dim) {
      for (int64_t i = 0; i < slice; i++, input += in_strides[d])
        Fill<NeedNormalize>(*output++, *input, mean++, inv_stddev++);
      fill_values += slice;
    } else {
      for (int64_t i = 0; i < slice; i++, input += in_strides[d])
        Fill<NeedNormalize>(*output++, *input, mean, inv_stddev);
    }

    // out of bounds (right side)
    if (pad_after > 0) {
      if (HasChannels && d == channel_dim) {
        for (int64_t i = 0; i < pad_after; i++, input += in_strides[d])
          *output++ = *fill_values++;
      } else {
        for (int64_t i = 0; i < pad_after; i++, input += in_strides[d])
          *output++ = *fill_values;
      }
    }
  }
}

template <bool NeedNormalize, bool HasChannels, bool OutOfBounds, typename OutputType,
          typename InputType, int DimsLeft>
void SliceFlipNormalizePermutePadKernelImpl(
    OutputType *output, const InputType *input, const int64_t *in_strides,
    const int64_t *out_strides, const int64_t *anchor, const int64_t *in_shape,
    const int64_t *out_shape, const OutputType *fill_values, const float *mean,
    const float *inv_stddev,
    int channel_dim,  // negative if no channel dim or already processed
    std::integral_constant<int, DimsLeft>) {
  constexpr int d = 0;
  int64_t pad_before, slice, pad_after;
  std::tie(pad_before, slice, pad_after) = CalcPadCopyExtents(anchor[d], in_shape[d], out_shape[d]);
  if (in_strides[d] < 0) std::swap(pad_before, pad_after);

  // out of bounds (left side)
  if (pad_before > 0) {
    if (HasChannels && d == channel_dim) {
      for (int64_t i = 0; i < pad_before; i++, output += out_strides[d], input += in_strides[d])
        SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, HasChannels, true>(
            output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
            fill_values++, mean++, inv_stddev++, channel_dim - 1,
            std::integral_constant<int, DimsLeft - 1>());
    } else {
      for (int64_t i = 0; i < pad_before; i++, output += out_strides[d], input += in_strides[d])
        SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, HasChannels, true>(
            output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
            fill_values, mean, inv_stddev, channel_dim - 1,
            std::integral_constant<int, DimsLeft - 1>());
    }
  }

  // within input bounds
  if (HasChannels && d == channel_dim) {
    for (int64_t i = 0; i < slice; i++, output += out_strides[d], input += in_strides[d])
      SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, HasChannels, OutOfBounds>(
          output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
          fill_values++, mean++, inv_stddev++, channel_dim - 1,
          std::integral_constant<int, DimsLeft - 1>());
  } else {
    for (int64_t i = 0; i < slice; i++, output += out_strides[d], input += in_strides[d])
      SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, HasChannels, OutOfBounds>(
          output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
          fill_values, mean, inv_stddev, channel_dim - 1,
          std::integral_constant<int, DimsLeft - 1>());
  }

  // out of bounds (right side)
  if (pad_after > 0) {
    if (HasChannels && d == channel_dim) {
      for (int64_t i = 0; i < pad_after; i++, output += out_strides[d], input += in_strides[d])
        SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, HasChannels, true>(
            output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
            fill_values++, mean++, inv_stddev++, channel_dim - 1,
            std::integral_constant<int, DimsLeft - 1>());
    } else {
      for (int64_t i = 0; i < pad_after; i++, output += out_strides[d], input += in_strides[d])
        SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, HasChannels, true>(
            output, input, in_strides + 1, out_strides + 1, anchor + 1, in_shape + 1, out_shape + 1,
            fill_values, mean, inv_stddev, channel_dim - 1,
            std::integral_constant<int, DimsLeft - 1>());
    }
  }
}

}  // namespace detail

template <int Dims, typename OutputType, typename InputType>
void SliceFlipNormalizePermutePadKernel(
    OutputType *output, const InputType *input, const TensorShape<Dims> &in_strides,
    const TensorShape<Dims> &out_strides, const TensorShape<Dims> &anchor,
    const TensorShape<Dims> &in_shape, const TensorShape<Dims> &out_shape,
    const OutputType *fill_values = nullptr, const float *mean = nullptr,
    const float *inv_stddev = nullptr,
    int channel_dim = -1) {  // negative if no channel dim or already processed
  bool need_pad = NeedPad(Dims, anchor.data(), in_shape.data(), out_shape.data());
  bool has_channels = channel_dim >= 0;
  bool need_normalize = (mean != nullptr && inv_stddev != nullptr);
  // Convert switch argument to `int` to avoid compiler warning about unreachable case label
  BOOL_SWITCH(need_normalize, NeedNormalize, (
    BOOL_SWITCH(has_channels, HasChannels, (
      if (need_pad) {
        constexpr bool OutOfBounds = false;
        detail::SliceFlipNormalizePermutePadKernelImpl<NeedNormalize, HasChannels, OutOfBounds>(
            output, input, in_strides.data(), out_strides.data(), anchor.data(), in_shape.data(),
            out_shape.data(), fill_values, mean, inv_stddev, channel_dim,
            std::integral_constant<int, Dims>());
      } else {
        detail::SliceFlipNormalizePermuteKernelImpl<NeedNormalize, HasChannels>(
            output, input, in_strides.data(), out_strides.data(), anchor.data(), in_shape.data(),
            out_shape.data(), mean, inv_stddev, channel_dim,
            std::integral_constant<int, Dims>());
      }
    ));  // NOLINT
  ));  // NOLINT
}


template <typename ExecutionEngine, int Dims, typename OutputType, typename InputType>
DLL_LOCAL  // workaround for GCC bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
void SliceFlipNormalizePermutePadKernel(
        ExecutionEngine &exec_engine, OutputType *output, const InputType *input,
        const detail::SliceFlipNormalizePermutePadProcessedArgs<Dims> &args,
        const SmallVector<OutputType, 8> &fill_values, int min_blk_sz = kSliceMinBlockSize,
        int req_nblocks = -1) {
  // Parallelize
  std::array<int, Dims> split_factor;
  uint64_t skip_dim_mask = args.channel_dim >= 0 ? 1_u64 << args.channel_dim : 0;
  int nblocks = split_shape(split_factor, args.out_shape,
                            req_nblocks > 0 ? req_nblocks : exec_engine.NumThreads() * 8,
                            min_blk_sz, skip_dim_mask);
  if (nblocks == 1) {
    exec_engine.AddWork([=](int) {
      SliceFlipNormalizePermutePadKernel(output, input, args.in_strides, args.out_strides,
                                         args.anchor, args.in_shape, args.out_shape,
                                         GetPtr<OutputType>(fill_values),
                                         GetPtr<float>(args.mean),
                                         GetPtr<float>(args.inv_stddev),
                                         args.channel_dim);
    }, kSliceFlipNormPermCost * volume(args.out_shape), false);
    return;
  }

  TensorShape<Dims> start;  // zero-filled
  const auto& end = args.out_shape;

  ForEachBlock(
    start, end, split_factor, 0, LastSplitDim(split_factor),
    [&](const TensorShape<Dims> &blk_start, const TensorShape<Dims> &blk_end) {
      auto output_ptr = output;
      auto input_ptr = input;
      TensorShape<Dims> blk_anchor;
      TensorShape<Dims> blk_shape;
      for (int d = 0; d < Dims; d++) {
        output_ptr += blk_start[d] * args.out_strides[d];
        input_ptr += blk_start[d] * args.in_strides[d];
        blk_shape[d] = blk_end[d] - blk_start[d];
        blk_anchor[d] = args.anchor[d] + blk_start[d];
      }

      exec_engine.AddWork([=](int) {
        SliceFlipNormalizePermutePadKernel(output_ptr, input_ptr, args.in_strides, args.out_strides,
                                           blk_anchor, args.in_shape, blk_shape,
                                           GetPtr<OutputType>(fill_values),
                                           GetPtr<float>(args.mean),
                                           GetPtr<float>(args.inv_stddev),
                                           args.channel_dim);
      }, kSliceFlipNormPermCost * volume(blk_shape), false);
    });
}

template <int Dims, typename OutputType, typename InputType>
void SliceFlipNormalizePermutePadKernel(
        SequentialExecutionEngine &exec_engine,
        OutputType *output, const InputType *input,
        const detail::SliceFlipNormalizePermutePadProcessedArgs<Dims> &args,
        const SmallVector<OutputType, 8> &fill_values,
        int /* min_blk_sz */ = -1, int /* req_nblocks */ = -1) {
  (void)exec_engine;
  SliceFlipNormalizePermutePadKernel(output, input, args.in_strides, args.out_strides, args.anchor,
                                     args.in_shape, args.out_shape,
                                     GetPtr<OutputType>(fill_values),
                                     GetPtr<float>(args.mean),
                                     GetPtr<float>(args.inv_stddev),
                                     args.channel_dim);
}


template <typename OutputType, typename InputType, int Dims>
class SliceFlipNormalizePermutePadCpu {
 public:
  static_assert(Dims >= 0, "Dims must be >= 0");

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

  template <typename ExecutionEngine>
  void Schedule(KernelContext &context,
                const OutTensorCPU<OutputType, Dims> &out,
                const InTensorCPU<InputType, Dims> &in,
                const Args &orig_args,
                ExecutionEngine &exec_engine,
                int min_blk_sz = 16000, int req_nblocks = -1) {
    auto args = detail::ProcessArgs(orig_args, in.shape);
    SmallVector<OutputType, 8> fill_values;
    for (auto value : args.fill_values)
      fill_values.push_back(static_cast<OutputType>(value));

    SliceFlipNormalizePermutePadKernel(exec_engine, out.data, in.data + args.input_offset, args,
                                       fill_values, min_blk_sz, req_nblocks);
  }

  /**
   * @brief Run the kernel
   */
  void Run(KernelContext &context,
           OutTensorCPU<OutputType, Dims> out,
           InTensorCPU<InputType, Dims> in,
           const Args &orig_args) {
    SequentialExecutionEngine engine;
    Schedule(context, out, in, orig_args, engine);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_CPU_H_
