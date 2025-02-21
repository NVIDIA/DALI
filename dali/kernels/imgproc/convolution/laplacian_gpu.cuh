// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_GPU_CUH_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_GPU_CUH_

#include <memory>
#include <utility>
#include <vector>

#include "dali/core/dev_buffer.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/cast_gpu.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/convolution_gpu.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_gpu.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/util/operator_impl_utils.h"


namespace dali {
namespace kernels {

namespace laplacian {

/**
 * @brief Computes convolution to obtain partial derivative in one of the dimensions.
 * Convolution consits of `axes` windows, each to convolve along one dimension of the input data,
 * where `deriv_axis`-th window is supposed to compute partial derivative along that axis,
 * whereas the remaining windows should perform smoothing. If no smoothing is necessary in
 * the whole batch, you can prevent smoothing convolutions from running by passing empty lists for
 * `window_sizes[i]` such that `i != deriv_axis`.
 */
template <typename Out, typename In, typename W, int axes, int deriv_axis, bool has_channels,
          bool is_sequence>
struct PartialDerivGpu {
  using MultiDimConv = SeparableConvolutionGpu<Out, In, W, axes, has_channels, is_sequence>;
  static constexpr int ndim = MultiDimConv::ndim;
  static constexpr int sequence_axes = MultiDimConv::sequence_axes;
  using SingleDimConv = ConvolutionGpu<Out, In, W, ndim, sequence_axes + deriv_axis, has_channels>;

  bool HasSmoothing(const std::array<TensorListShape<1>, axes>& window_sizes) {
    for (int axis = 0; axis < axes; axis++) {
      if (axis != deriv_axis && window_sizes[axis].num_samples() > 0) {
        return true;
      }
    }
    return false;
  }

  /**
   * @param ctx             Kernel context, used for scratch-pad.
   * @param in_shape        List of input shapes, used by underlaying convolution kernels to infer
   *                        intermediate buffer sizes.
   * @param window_sizes    For given `i`, `window_sizes[i]` contains per-sample window sizes
   *                        to be applied in a convolution along `i-th` axis. The length of
   *                        `window_sizes[deriv_axis]` must be equal to the input batch size.
   *                        Lists for other axes must either all have length equal to the input
   *                        batch size or all be empty. In the latter case, smoothing convolutions
   *                        will be omitted, i.e. only one convolution, along `deriv_axis`
   *                        will be applied.
   */
  KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape,
                           const std::array<TensorListShape<1>, axes>& window_sizes) {
    has_smoothing_ = HasSmoothing(window_sizes);
    if (has_smoothing_) {
      if (!multi_dim_impl_) {
        multi_dim_impl_ = std::make_unique<MultiDimConv>();
      }
      return multi_dim_impl_->Setup(ctx, in_shape, window_sizes, true);
    }
    if (!single_dim_impl_) {
      single_dim_impl_ = std::make_unique<SingleDimConv>();
    }
    return single_dim_impl_->Setup(ctx, in_shape, window_sizes[deriv_axis]);
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const std::array<TensorListView<StorageCPU, const W, 1>, axes>& windows,
           const ConvEpilogue& conv_epilogue) {
    if (has_smoothing_) {
      assert(multi_dim_impl_);
      multi_dim_impl_->Run(ctx, out, in, windows, {}, conv_epilogue);
    } else {
      assert(single_dim_impl_);
      single_dim_impl_->Run(ctx, out, in, windows[deriv_axis], {}, conv_epilogue);
    }
  }

  bool has_smoothing_;
  std::unique_ptr<MultiDimConv> multi_dim_impl_;
  std::unique_ptr<SingleDimConv> single_dim_impl_;
};

/**
 * @brief Provides Laplacian specializations for 2 and 3 dimensional data.
 */
template <typename Out, typename In, typename W, int axes, bool has_channels, bool is_sequence>
struct LaplacianGpuBase;

template <typename Out, typename In, typename W, bool has_channels, bool is_sequence>
struct LaplacianGpuBase<Out, In, W, 2, has_channels, is_sequence> {
  static constexpr int axes = 2;
  using DyKernel = PartialDerivGpu<Out, In, W, axes, 0, has_channels, is_sequence>;
  using DxKernel = PartialDerivGpu<Out, In, W, axes, 1, has_channels, is_sequence>;
  static constexpr int ndim = DxKernel::ndim;  // DyKernel::ndim is the same

  KernelRequirements Setup(
      KernelContext& ctx, const TensorListShape<ndim>& in_shape,
      const std::array<std::array<TensorListShape<1>, axes>, axes>& window_sizes) {
    KernelRequirements req;
    req.output_shapes.push_back(in_shape);

    auto req_dy = dy_kernel_.Setup(ctx, in_shape, window_sizes[0]);
    auto req_dx = dx_kernel_.Setup(ctx, in_shape, window_sizes[1]);
    return req;
  }

  void Run(
      KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
      const TensorListView<StorageGPU, const In, ndim>& in,
      const std::array<std::array<TensorListView<StorageCPU, const W, 1>, axes>, axes>& windows,
      std::array<span<const float>, axes> scale) {
    {
      KernelContext sub_ctx = ctx;
      kernels::DynamicScratchpad dyn_scratchpad(AccessOrder(sub_ctx.gpu.stream));
      sub_ctx.scratchpad = &dyn_scratchpad;
      dy_kernel_.Run(sub_ctx, out, in, windows[0], {scale[0]});
    }
    {
      KernelContext sub_ctx = ctx;
      kernels::DynamicScratchpad dyn_scratchpad(AccessOrder(sub_ctx.gpu.stream));
      sub_ctx.scratchpad = &dyn_scratchpad;
      dx_kernel_.Run(sub_ctx, out, in, windows[1], {scale[1], 1.f});
    }
  }

  DyKernel dy_kernel_;
  DxKernel dx_kernel_;
};

template <typename Out, typename In, typename W, bool has_channels, bool is_sequence>
struct LaplacianGpuBase<Out, In, W, 3, has_channels, is_sequence> {
  static constexpr int axes = 3;
  using DzKernel = PartialDerivGpu<Out, In, W, axes, 0, has_channels, is_sequence>;
  using DyKernel = PartialDerivGpu<Out, In, W, axes, 1, has_channels, is_sequence>;
  using DxKernel = PartialDerivGpu<Out, In, W, axes, 2, has_channels, is_sequence>;
  static constexpr int ndim = DxKernel::ndim;

  KernelRequirements Setup(
      KernelContext& ctx, const TensorListShape<ndim>& in_shape,
      const std::array<std::array<TensorListShape<1>, axes>, axes>& window_sizes) {
    KernelRequirements req;
    req.output_shapes.push_back(in_shape);

    auto req_dz = dz_kernel_.Setup(ctx, in_shape, window_sizes[0]);
    auto req_dy = dy_kernel_.Setup(ctx, in_shape, window_sizes[1]);
    auto req_dx = dx_kernel_.Setup(ctx, in_shape, window_sizes[2]);
    return req;
  }

  void Run(
      KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
      const TensorListView<StorageGPU, const In, ndim>& in,
      const std::array<std::array<TensorListView<StorageCPU, const W, 1>, axes>, axes>& windows,
      std::array<span<const float>, axes> scale) {
    {
      KernelContext sub_ctx = ctx;
      kernels::DynamicScratchpad dyn_scratchpad(AccessOrder(sub_ctx.gpu.stream));
      sub_ctx.scratchpad = &dyn_scratchpad;
      dz_kernel_.Run(sub_ctx, out, in, windows[0], {scale[0]});
    }
    {
      KernelContext sub_ctx = ctx;
      kernels::DynamicScratchpad dyn_scratchpad(AccessOrder(sub_ctx.gpu.stream));
      sub_ctx.scratchpad = &dyn_scratchpad;
      dy_kernel_.Run(sub_ctx, out, in, windows[1], {scale[1], 1.f});
    }
    {
      KernelContext sub_ctx = ctx;
      kernels::DynamicScratchpad dyn_scratchpad(AccessOrder(sub_ctx.gpu.stream));
      sub_ctx.scratchpad = &dyn_scratchpad;
      dx_kernel_.Run(sub_ctx, out, in, windows[2], {scale[2], 1.f});
    }
  }

  DzKernel dz_kernel_;
  DyKernel dy_kernel_;
  DxKernel dx_kernel_;
};


/**
 * @brief Laplacian kernel.
 * Provides separate specialization for 1 dimensional data. For other cases
 * there are two specializations depending on whether ``Intermediate`` is the same as ``Out``:
 * if not, the extra intermediate buffer is used to accumulate partial derivatives and then
 * the result is cast and moved to the output TensorListView.
 *
 * For `axes` dimensional input data, there will be computed and summed `axes` partial derivatives,
 * one in each direction. The `i-th` partial derivative is computed by performing
 * a `axes`-dimensional separable convolution, where the `i-th` window is responsible for actual
 * derivative approximation and the remaining windows provide smoothing.
 * In total, there are ``axes * axes`` windows involved in computing laplacian.
 *
 * Dummy type is used with enable_if_t to disambiguate template specializations.
 */
template <typename Intermediate, typename Out, typename In, typename W, int axes, bool has_channels,
          bool is_sequence, typename Dummy = void>
struct LaplacianGpu;

template <typename Out, typename In, typename W, int axes, bool has_channels, bool is_sequence>
struct LaplacianGpu<Out, Out, In, W, axes, has_channels, is_sequence, std::enable_if_t<axes != 1>>
    : LaplacianGpuBase<Out, In, W, axes, has_channels, is_sequence> {};

template <typename Intermediate, typename Out, typename In, typename W, bool has_channels,
          bool is_sequence>
struct LaplacianGpu<Intermediate, Out, In, W, 1, has_channels, is_sequence> {
  static constexpr int axes = 1;
  using DxKernel = SeparableConvolutionGpu<Out, In, W, axes, has_channels, is_sequence>;
  static constexpr int ndim = DxKernel::ndim;

  KernelRequirements Setup(
      KernelContext& ctx, const TensorListShape<ndim>& in_shape,
      const std::array<std::array<TensorListShape<1>, axes>, axes>& window_sizes) {
    return dx_kernel_.Setup(ctx, in_shape, window_sizes[0]);
  }

  void Run(
      KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
      const TensorListView<StorageGPU, const In, ndim>& in,
      const std::array<std::array<TensorListView<StorageCPU, const W, 1>, axes>, axes>& windows,
      std::array<span<const float>, axes> scale) {
    return dx_kernel_.Run(ctx, out, in, windows[0], {}, {scale[0]});
  }

  DxKernel dx_kernel_;
};

template <typename Intermediate, typename Out, typename In, typename W, int axes, bool has_channels,
          bool is_sequence>
struct LaplacianGpu<Intermediate, Out, In, W, axes, has_channels, is_sequence,
                    std::enable_if_t<!std::is_same<Intermediate, Out>::value && axes != 1>>
    : LaplacianGpuBase<Intermediate, In, W, axes, has_channels, is_sequence> {
  using Base = LaplacianGpuBase<Intermediate, In, W, axes, has_channels, is_sequence>;
  using Base::ndim;

  void Run(
      KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
      const TensorListView<StorageGPU, const In, ndim>& in,
      const std::array<std::array<TensorListView<StorageCPU, const W, 1>, axes>, axes>& windows,
      std::array<span<const float>, axes> scale) {
    auto intermediate =
        ctx.scratchpad->AllocTensorList<mm::memory_kind::device, Intermediate, ndim>(in.shape);
    Base::Run(ctx, intermediate, in, windows, scale);
    kernels::cast::CastGPU<Out, Intermediate> cast;
    cast.Run(ctx, flatten(out), flatten(intermediate));
  }
};

}  // namespace laplacian

template <typename Out, typename In, typename W, int axes, bool has_channels, bool is_sequence>
using LaplacianGpu = laplacian::LaplacianGpu<decltype(std::declval<W>() * std::declval<In>()), Out,
                                             In, W, axes, has_channels, is_sequence>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_GPU_CUH_
