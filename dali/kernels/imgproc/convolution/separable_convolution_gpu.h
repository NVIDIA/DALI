// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_CONVOLUTION_GPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_CONVOLUTION_GPU_H_

#include "dali/core/convert.h"
#include "dali/core/format.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/convolution_gpu.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/scratch.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {
namespace kernels {



/**
 * @brief Apply convolution in all spatial axes, starting from the innermost to outermost.
 *        If channel axis is pressent, the convolution is not applied there.
 *        If it is marked as sequence, the outermost dimension denotes frames and
 *        convolution is not applied to it.
 *
 * Sequence convolutions require one less window to be specified
 *
 * `W` is currently assumed to be float
 *
 * `windows` and `scales` are specified per axis.
 *
 * Specialized for 1, 2 or 3 axes, to not go overboard with TMP for generic solutions
 *
 * Here be boilerplate.
 */
template <typename Out, typename In, typename W, int axes, bool has_channels = false,
          bool is_sequence = false>
struct SeparableConvolutionGpu;

template <typename Out, typename In, typename W, bool has_channels, bool is_sequence>
struct SeparableConvolutionGpu<Out, In, W, 1, has_channels, is_sequence> {
  static constexpr int axes = 1;
  static constexpr int sequence_axes = static_cast<int>(is_sequence);
  static constexpr int channel_axes = static_cast<int>(has_channels);
  static constexpr int ndim = sequence_axes + axes + channel_axes;

  KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape,
                           const std::array<TensorListShape<1>, axes>& window_sizes) {
    KernelRequirements req;
    req.output_shapes.push_back(in_shape);

    auto req_conv = conv_.Setup(ctx, in_shape, window_sizes[0]);

    req.scratch_sizes = AppendScratchSize(req.scratch_sizes, req_conv.scratch_sizes);

    return req;
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const std::array<TensorListView<StorageCPU, const W, 1>, axes>& windows,
           const std::array<span<const int>, 1> anchors = {},
           float scale = 1) {
    conv_.Run(ctx, out, in, windows[0], anchors[0], scale);
  }

  ConvolutionGpu<Out, In, W, ndim, sequence_axes + 0, has_channels> conv_;
};

template <typename Out, typename In, typename W, bool has_channels, bool is_sequence>
struct SeparableConvolutionGpu<Out, In, W, 2, has_channels, is_sequence> {
  static constexpr int axes = 2;
  static constexpr int sequence_axes = static_cast<int>(is_sequence);
  static constexpr int channel_axes = static_cast<int>(has_channels);
  static constexpr int ndim = sequence_axes + axes + channel_axes;
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());

  KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape,
                           const std::array<TensorListShape<1>, axes>& window_sizes) {
    KernelRequirements req;

    ScratchpadEstimator se;
    se.add<mm::memory_kind::device, Intermediate>(in_shape.num_elements());
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(in_shape);

    auto req_inner = conv_innermost_.Setup(ctx, in_shape, window_sizes[1]);
    auto req_outer = conv_outermost_.Setup(ctx, in_shape, window_sizes[0]);

    // Calculate max scratch memory required by sub-kernels
    sub_scratch_sizes_ = MaxScratchSize(req_inner.scratch_sizes, req_outer.scratch_sizes);
    req.scratch_sizes = AppendScratchSize(req.scratch_sizes, sub_scratch_sizes_);

    return req;
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const std::array<TensorListView<StorageCPU, const W, 1>, axes>& windows,
           const std::array<span<const int>, 2> anchors = {},
           float scale = 1) {
    auto *tmp = ctx.scratchpad->AllocateGPU<Intermediate>(in.shape.num_elements());

    auto intermediate = TensorListView<StorageGPU, Intermediate, ndim>(tmp, in.shape);

    // Prepare the scratchpad with all the remaining memory requested by sub-kernels
    // TODO(michalz): Get rid of this run-time memory kind dispatch
    PreallocatedScratchpad sub_scratch;
    for (size_t i = 0; i < sub_scratch_sizes_.size(); i++) {
      auto sz = sub_scratch_sizes_[i];
      auto kind_id = static_cast<mm::memory_kind_id>(i);
      sub_scratch.allocs[i] = BumpAllocator(
        static_cast<char*>(ctx.scratchpad->Alloc(kind_id, sz, 64)), sz);
    }

    KernelContext sub_ctx = ctx;
    sub_ctx.scratchpad = &sub_scratch;

    // Clear the scratchpad for sub-kernels to reuse memory
    conv_innermost_.Run(sub_ctx, intermediate, in, windows[1], anchors[1]);
    sub_scratch.Clear();
    conv_outermost_.Run(sub_ctx, out, intermediate, windows[0], anchors[0], scale);
  }

  scratch_sizes_t sub_scratch_sizes_;
  ConvolutionGpu<Intermediate, In, W, ndim, sequence_axes + 1, has_channels> conv_innermost_;
  ConvolutionGpu<Out, Intermediate, W, ndim, sequence_axes + 0, has_channels> conv_outermost_;
};

template <typename Out, typename In, typename W, bool has_channels, bool is_sequence>
struct SeparableConvolutionGpu<Out, In, W, 3, has_channels, is_sequence> {
  static constexpr int axes = 3;
  static constexpr int sequence_axes = static_cast<int>(is_sequence);
  static constexpr int channel_axes = static_cast<int>(has_channels);
  static constexpr int ndim = sequence_axes + axes + channel_axes;
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());
  static constexpr bool kUseOutAsIntermediate = sizeof(Intermediate) == sizeof(Out);

  KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape,
                           const std::array<TensorListShape<1>, axes>& window_sizes) {
    KernelRequirements req;

    ScratchpadEstimator se;
    int intermediate_count = kUseOutAsIntermediate ? 1 : 2;
    se.add<mm::memory_kind::device, Intermediate>(in_shape.num_elements() * intermediate_count);
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(in_shape);

    auto req_inner = conv_innermost_.Setup(ctx, in_shape, window_sizes[2]);
    auto req_middle = conv_middle_.Setup(ctx, in_shape, window_sizes[1]);
    auto req_outer = conv_outermost_.Setup(ctx, in_shape, window_sizes[0]);

    // Calculate max scratch memory required by sub-kernels
    sub_scratch_sizes_ = MaxScratchSize(req_inner.scratch_sizes, req_middle.scratch_sizes);
    sub_scratch_sizes_ = MaxScratchSize(sub_scratch_sizes_, req_outer.scratch_sizes);
    req.scratch_sizes = AppendScratchSize(req.scratch_sizes, sub_scratch_sizes_);

    return req;
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const std::array<TensorListView<StorageCPU, const W, 1>, axes>& windows,
           const std::array<span<const int>, 3> anchors = {},
           float scale = 1) {
    int intermediate_count = kUseOutAsIntermediate ? 1 : 2;
    auto* tmp = ctx.scratchpad->AllocateGPU<Intermediate>(
        in.shape.num_elements() * intermediate_count);
    TensorListView<StorageGPU, Intermediate, ndim> intermediate_inner, intermediate_outer;
    if (kUseOutAsIntermediate) {
      intermediate_inner = reinterpret<Intermediate>(out, in.shape);
      intermediate_outer = {tmp, in.shape};
    } else {
      intermediate_inner = {tmp, in.shape};
      intermediate_outer = {tmp + in.shape.num_elements(), in.shape};
    }

    // Prepare the scratchpad with all the remaining memory requested by sub-kernels
    // TODO(michalz): Get rid of this run-time memory kind dispatch
    PreallocatedScratchpad sub_scratch;
    for (size_t i = 0; i < sub_scratch_sizes_.size(); i++) {
      auto sz = sub_scratch_sizes_[i];
      auto kind_id = static_cast<mm::memory_kind_id>(i);
      sub_scratch.allocs[i] = BumpAllocator(
        static_cast<char*>(ctx.scratchpad->Alloc(kind_id, sz, 64)), sz);
    }

    KernelContext sub_ctx = ctx;
    sub_ctx.scratchpad = &sub_scratch;

    // Clear the scratchpad for sub-kernels to reuse memory
    conv_innermost_.Run(sub_ctx, intermediate_inner, in, windows[2], anchors[2]);
    sub_scratch.Clear();
    conv_middle_.Run(sub_ctx, intermediate_outer, intermediate_inner, windows[1], anchors[1]);
    sub_scratch.Clear();
    conv_outermost_.Run(sub_ctx, out, intermediate_outer, windows[0], anchors[0], scale);
  }

  scratch_sizes_t sub_scratch_sizes_;
  ConvolutionGpu<Intermediate, In, W, ndim, sequence_axes + 2, has_channels> conv_innermost_;
  ConvolutionGpu<Intermediate, Intermediate, W, ndim, sequence_axes + 1, has_channels> conv_middle_;
  ConvolutionGpu<Out, Intermediate, W, ndim, sequence_axes + 0, has_channels> conv_outermost_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_CONVOLUTION_GPU_H_
