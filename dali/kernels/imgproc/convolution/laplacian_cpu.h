// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_CPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_CPU_H_

#include <memory>

#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_cpu.h"
#include "dali/kernels/imgproc/convolution/convolution_cpu.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/scratch.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {
namespace kernels {

namespace laplacian {

using namespace conv_transform;  // NOLINT

/**
 * @brief Computes convolution to obtain partial derivative in one of the dimensions.
 * If it is OpenCV style convolution that applies smoothing in perpendicular dimensions,
 * SeparableConvolution kernel is used to compute multidimensional convolution,
 * otherwise (if window sizes in non derivative dimensions are one) skips unnecessary
 * convolutions, performing only single one dimensional convolution with derivative kernel.
 */
template <typename Out, typename In, typename W, int axes, int deriv_axis,
          bool has_channels = false, typename T = conv_transform::TransScaleSat<Out, W>>
struct Convolution {
  using MultiDimConv = SeparableConvolutionCpu<Out, In, W, axes, has_channels, T>;
  static constexpr int ndim = MultiDimConv::ndim;
  using SingleDimConv = ConvolutionCpu<Out, In, W, ndim, deriv_axis, has_channels, T>;

  bool IsNoSmooting(const std::array<int, axes>& window_sizes) {
    for (int axis = 0; axis < axes; axis++) {
      if (axis != deriv_axis && window_sizes[axis] != 1) {
        return false;
      }
    }
    return true;
  }

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<int, axes>& window_sizes) {
    no_smoothing_ = IsNoSmooting(window_sizes);
    if (no_smoothing_) {
      if (!single_dim_impl_) {
        multi_dim_impl_.reset();
        single_dim_impl_ = std::make_unique<SingleDimConv>();
      }
      return single_dim_impl_->Setup(ctx, in_shape, window_sizes[deriv_axis]);
    }
    if (!multi_dim_impl_) {
      single_dim_impl_.reset();
      multi_dim_impl_ = std::make_unique<MultiDimConv>();
    }
    return multi_dim_impl_->Setup(ctx, in_shape, window_sizes);
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> &out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<TensorView<StorageCPU, const W, 1>, axes>& windows,
           const T& transform = {}) {
    if (no_smoothing_) {
      single_dim_impl_->Run(ctx, out, in, windows[deriv_axis], transform);
    } else {
      multi_dim_impl_->Run(ctx, out, in, windows, transform);
    }
  }

  bool no_smoothing_;
  std::unique_ptr<MultiDimConv> multi_dim_impl_;
  std::unique_ptr<SingleDimConv> single_dim_impl_;
};


/**
 * @brief Provides Laplacian specializations for 2 and 3 dimensional data.
 *  Run methods expect additional ``acc`` buffer that will be used to accumulate
 *  partial derivatives. If ``Intermediate`` and ``Out`` are the same type,
 *  the `acc` and ``out`` can be the same tensor.
 */
template <typename T, typename Intermediate, typename Out, typename In, typename W, int axes,
          bool has_channels>
struct LaplacianCPUBase;

template <typename T, typename Intermediate, typename Out, typename In, typename W,
          bool has_channels>
struct LaplacianCPUBase<T, Intermediate, Out, In, W, 2, has_channels> {
  static constexpr int axes = 2;
  using DxKernel = Convolution<Intermediate, In, W, axes, 0, has_channels,
                               TransScaleSat<Intermediate, W>>;
  using DyKernel = Convolution<Out, In, W, axes, 1, has_channels, T>;
  static constexpr int ndim = DyKernel::ndim;  // Dx ndim is the same

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<std::array<int, axes>, axes>& window_sizes) {
    KernelRequirements req;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in_shape));

    auto req_dx = sobel_dx_.Setup(ctx, in_shape, window_sizes[0]);
    auto req_dy = sobel_dy_.Setup(ctx, in_shape, window_sizes[1]);

    // Calculate max scratch memory required by sub-kernels
    sub_scratch_sizes_ = MaxScratchSize(req_dx.scratch_sizes, req_dy.scratch_sizes);
    req.scratch_sizes = sub_scratch_sizes_;

    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> &out,
           const TensorView<StorageCPU, Intermediate, ndim> &acc,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes>& windows,
           const std::array<float, axes>& scale, const T& transform) {
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
    sobel_dx_.Run(sub_ctx, acc, in, windows[0], scale[0]);
    sub_scratch.Clear();
    sobel_dy_.Run(sub_ctx, out, in, windows[1], transform);
  }

  scratch_sizes_t sub_scratch_sizes_;
  DxKernel sobel_dx_;
  DyKernel sobel_dy_;
};

template <typename T, typename Intermediate, typename Out, typename In, typename W,
          bool has_channels>
struct LaplacianCPUBase<T, Intermediate, Out, In, W, 3, has_channels> {
  static constexpr int axes = 3;
  using DxKernel = Convolution<Intermediate, In, W, axes, 0, has_channels,
                               TransScaleSat<Intermediate, W>>;
  using DyKernel = Convolution<Intermediate, In, W, axes, 1, has_channels,
                               TransScaleAddOutSat<Intermediate, W>>;
  using DzKernel = Convolution<Out, In, W, axes, 2, has_channels, T>;
  static constexpr int ndim = DzKernel::ndim;  // Dx and Dy ndim are the same

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<std::array<int, axes>, axes>& window_sizes) {
    KernelRequirements req;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in_shape));

    auto req_dx = sobel_dx_.Setup(ctx, in_shape, window_sizes[0]);
    auto req_dy = sobel_dy_.Setup(ctx, in_shape, window_sizes[1]);
    auto req_dz = sobel_dz_.Setup(ctx, in_shape, window_sizes[2]);

    // Calculate max scratch memory required by sub-kernels
    sub_scratch_sizes_ = MaxScratchSize(req_dx.scratch_sizes, req_dy.scratch_sizes);
    sub_scratch_sizes_ = MaxScratchSize(sub_scratch_sizes_, req_dz.scratch_sizes);
    req.scratch_sizes = sub_scratch_sizes_;

    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> &out,
           const TensorView<StorageCPU, Intermediate, ndim> &acc,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes>& windows,
           const std::array<float, axes>& scale, const T& transform) {
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
    sobel_dx_.Run(sub_ctx, acc, in, windows[0], scale[0]);
    sub_scratch.Clear();
    sobel_dy_.Run(sub_ctx, acc, in, windows[1], scale[1]);
    sub_scratch.Clear();
    sobel_dz_.Run(sub_ctx, out, in, windows[2], transform);
  }

  scratch_sizes_t sub_scratch_sizes_;
  DxKernel sobel_dx_;
  DyKernel sobel_dy_;
  DzKernel sobel_dz_;
};


/**
 * @brief Laplacian kernel. Provides separate specialization for 1 dimensional data. For other cases
 * there are two specializations depending on whether ``Intermediate`` is the same as ``Out``, if not
 * the extra intermediate buffer is used to accumulate partial derivatives.
 */
template <typename Intermediate, typename Out, typename In, typename W, int axes, bool has_channels,
          typename Dummy = void>
struct LaplacianCPU;

template <typename Intermediate, typename Out, typename In, typename W, int axes, bool has_channels>
struct LaplacianCPU<Intermediate, Out, In, W, axes, has_channels,
                    std::enable_if_t<!std::is_same<Intermediate, Out>::value && axes != 1>>
                    : LaplacianCPUBase<TransScaleAddBufferSat<Intermediate, Out, W>,
                                       Intermediate, Out, In, W, axes, has_channels>  {
  using Base = LaplacianCPUBase<TransScaleAddBufferSat<Intermediate, Out, W>, Intermediate,
                                Out, In, W, axes, has_channels>;
  using Base::ndim;

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<std::array<int, axes>, axes>& window_sizes) {
    auto req = Base::Setup(ctx, in_shape, window_sizes);
    ScratchpadEstimator se;
    se.add<mm::memory_kind::host, Intermediate>(volume(in_shape));
    req.scratch_sizes = AppendScratchSize(req.scratch_sizes, se.sizes);
    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> &out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes>& windows,
           const std::array<float, axes>& scale) {
    auto *tmp = ctx.scratchpad->AllocateHost<Intermediate>(volume(in.shape));
    auto acc = TensorView<StorageCPU, Intermediate, ndim>(tmp, in.shape);
    Base::Run(ctx, out, acc, in, windows, scale, {tmp, scale[axes - 1]});
  }
};

template <typename Out, typename In, typename W, int axes, bool has_channels>
struct LaplacianCPU<Out, Out, In, W, axes, has_channels, std::enable_if_t<axes != 1>>
                    : LaplacianCPUBase<TransScaleAddOutSat<Out, W>, Out, Out, In, W,
                                       axes, has_channels> {
  using Base = LaplacianCPUBase<TransScaleAddOutSat<Out, W>, Out, Out, In, W, axes, has_channels>;
  using Base::ndim;

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> &out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes>& windows,
           const std::array<float, axes>& scale) {
    Base::Run(ctx, out, out, in, windows, scale, scale[axes - 1]);
  }
};

template <typename Intermediate, typename Out, typename In, typename W, bool has_channels>
struct LaplacianCPU<Intermediate, Out, In, W, 1, has_channels> {
  static constexpr int axes = 1;
  using ConvKernel = SeparableConvolutionCpu<Out, In, W, axes, has_channels,
                                             TransScaleSat<Out, W>>;
  static constexpr int ndim = ConvKernel::ndim;

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<std::array<int, axes>, axes>& window_sizes) {
    return conv_.Setup(ctx, in_shape, window_sizes[0]);
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> &out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes>& windows,
           const std::array<float, axes>& scale) {
    return conv_.Run(ctx, out, in, windows[0], scale[0]);
  }

  ConvKernel conv_;
};

}  // namespace laplacian

template <typename Out, typename In, typename W, int axes, bool has_channels>
using LaplacianCPU = laplacian::LaplacianCPU<decltype(std::declval<W>() * std::declval<In>()),
                                             Out, In, W, axes, has_channels>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_CPU_H_
