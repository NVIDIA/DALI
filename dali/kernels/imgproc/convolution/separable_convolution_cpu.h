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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_CONVOLUTION_CPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_CONVOLUTION_CPU_H_

#include "dali/core/convert.h"
#include "dali/core/format.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/convolution_cpu.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/scratch.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {
namespace kernels {

/**
 * @brief Apply convolution in all spatial axes, starting from the innermost to outermost.
 *        If channel axis is pressent, the convolution is not applied there.
 *
 * If `Out` is same as `W` the intermediate stages are written to out,
 * otherwise the temporary buffer of `W` is required and allocated in scratchpad
 *
 * `W` is currently assumed to be float
 *
 * `windows` and `scales` are specified per axis.
 *
 * Specialized for 1, 2 or 3 axes, to not go overboard with TMP for generic solutions
 *
 * N.B. For more dimension, fusing a permute step when writing the result
 * could allow for processing all steps with innermost, contiguous dimension.
 * For example DHWC->DWHC->HWDC->DHWC, while applying convolutions for W, H, D respectively.
 * This might be faster, but vectorization on outer dims would still probably win.
 */
template <typename Out, typename In, typename W, int axes, bool has_channels = false>
struct SeparableConvolutionCpu;

template <typename Out, typename In, typename W, bool has_channels>
struct SeparableConvolutionCpu<Out, In, W, 1, has_channels> {
  static constexpr int axes = 1;
  static constexpr int ndim = has_channels ? 2 : 1;

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<int, axes>& window_sizes) {
    KernelRequirements req;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in_shape));

    auto req_conv = conv_.Setup(ctx, in_shape, window_sizes[0]);

    req.scratch_sizes = AppendScratchSize(req.scratch_sizes, req_conv.scratch_sizes);

    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> &out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<TensorView<StorageCPU, const W, 1>, axes>& windows,
           W scale = 1) {
    conv_.Run(ctx, out, in, windows[0], scale);
  }

  ConvolutionCpu<Out, In, W, ndim, 0, has_channels> conv_;
};

template <typename Out, typename In, typename W, bool has_channels>
struct SeparableConvolutionCpu<Out, In, W, 2, has_channels> {
  static constexpr int axes = 2;
  static constexpr int ndim = has_channels ? 3 : 2;
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<int, axes>& window_sizes) {
    KernelRequirements req;

    ScratchpadEstimator se;
    se.add<mm::memory_kind::host, Intermediate>(volume(in_shape));
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in_shape));

    auto req_inner = conv_innermost_.Setup(ctx, in_shape, window_sizes[1]);
    auto req_outer = conv_outermost_.Setup(ctx, in_shape, window_sizes[0]);

    // Calculate max scratch memory required by sub-kernels
    sub_scratch_sizes_ = MaxScratchSize(req_inner.scratch_sizes, req_outer.scratch_sizes);
    req.scratch_sizes = AppendScratchSize(req.scratch_sizes, sub_scratch_sizes_);

    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> &out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<TensorView<StorageCPU, const W, 1>, axes>& windows,
           W scale = 1) {
    auto *tmp = ctx.scratchpad->AllocateHost<Intermediate>(volume(in.shape));
    auto intermediate = TensorView<StorageCPU, Intermediate, ndim>(tmp, in.shape);

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
    conv_innermost_.Run(sub_ctx, intermediate, in, windows[1]);
    sub_scratch.Clear();
    conv_outermost_.Run(sub_ctx, out, intermediate, windows[0], scale);
  }

  scratch_sizes_t sub_scratch_sizes_;
  ConvolutionCpu<Intermediate, In, W, ndim, 1, has_channels> conv_innermost_;
  ConvolutionCpu<Out, Intermediate, W, ndim, 0, has_channels> conv_outermost_;
};

template <typename Out, typename In, typename W, bool has_channels>
struct SeparableConvolutionCpu<Out, In, W, 3, has_channels> {
  static constexpr int axes = 3;
  static constexpr int ndim = has_channels ? 4 : 3;
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<int, axes>& window_sizes) {
    KernelRequirements req;

    ScratchpadEstimator se;
    se.add<mm::memory_kind::host, Intermediate>(volume(in_shape));
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in_shape));

    auto req_inner = conv_innermost_.Setup(ctx, in_shape, window_sizes[2]);
    auto req_middle = conv_middle_.Setup(ctx, in_shape, window_sizes[1]);
    auto req_outer = conv_outermost_.Setup(ctx, in_shape, window_sizes[0]);

    // Calculate max scratch memory required by sub-kernels
    sub_scratch_sizes_ = MaxScratchSize(req_inner.scratch_sizes, req_middle.scratch_sizes);
    sub_scratch_sizes_ = MaxScratchSize(sub_scratch_sizes_, req_outer.scratch_sizes);
    req.scratch_sizes = AppendScratchSize(req.scratch_sizes, sub_scratch_sizes_);

    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> &out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<TensorView<StorageCPU, const W, 1>, axes>& windows,
           W scale = 1) {
    auto* tmp = ctx.scratchpad->AllocateHost<Intermediate>(volume(in.shape));
    auto intermediate = TensorView<StorageCPU, Intermediate, ndim>(tmp, in.shape);

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
    conv_innermost_.Run(sub_ctx, intermediate, in, windows[2]);
    sub_scratch.Clear();
    conv_middle_.Run(sub_ctx, intermediate, intermediate, windows[1]);
    sub_scratch.Clear();
    conv_outermost_.Run(sub_ctx, out, intermediate, windows[0], scale);
  }

  scratch_sizes_t sub_scratch_sizes_;
  ConvolutionCpu<Intermediate, In, W, ndim, 2, has_channels> conv_innermost_;
  ConvolutionCpu<Intermediate, Intermediate, W, ndim, 1, has_channels> conv_middle_;
  ConvolutionCpu<Out, Intermediate, W, ndim, 0, has_channels> conv_outermost_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_CONVOLUTION_CPU_H_
