// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/imgproc/convolution/convolution_cpu.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_cpu.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/scratch.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {
namespace kernels {

namespace laplacian {

/**
 * @brief Computes convolution to obtain partial derivative in one of the dimensions.
 * Convolution consits of `axes` windows, each to convolve along one dimension of the input data,
 * where `deriv_axis`-th window is supposed to compute partial derivative along that axis,
 * whereas the remaining windows should perform smoothing. It is assumed that smoothing
 * window of size 1 must be equal to `[1]`, this way, if window sizes in non-derivative directions
 * are one, the smoothing convolutions can be skipped and only a single one-dimensional
 * convolution in derivative direction is performed.
 */
template <typename Out, typename In, typename W, int axes, int deriv_axis,
          bool has_channels = false, typename T = conv_transform::TransScaleSat<Out, W>>
struct PartialDeriv {
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

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim>& out,
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
 *  the ``acc`` and ``out`` can be the same tensor.
 */
template <typename T, typename Intermediate, typename Out, typename In, typename W, int axes,
          bool has_channels>
struct LaplacianCpuBase;

template <typename T, typename Intermediate, typename Out, typename In, typename W,
          bool has_channels>
struct LaplacianCpuBase<T, Intermediate, Out, In, W, 2, has_channels> {
  static constexpr int axes = 2;
  using DyKernel = PartialDeriv<Intermediate, In, W, axes, 0, has_channels,
                                conv_transform::TransScaleSat<Intermediate, W>>;
  using DxKernel = PartialDeriv<Out, In, W, axes, 1, has_channels, T>;
  static constexpr int ndim = DyKernel::ndim;  // Dx ndim is the same

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<std::array<int, axes>, axes>& window_sizes) {
    KernelRequirements req;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in_shape));

    auto req_dy = dy_kernel_.Setup(ctx, in_shape, window_sizes[0]);
    auto req_dx = dx_kernel_.Setup(ctx, in_shape, window_sizes[1]);

    // Calculate max scratch memory required by sub-kernels
    sub_scratch_sizes_ = MaxScratchSize(req_dx.scratch_sizes, req_dy.scratch_sizes);
    req.scratch_sizes = sub_scratch_sizes_;

    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim>& out,
           const TensorView<StorageCPU, Intermediate, ndim>& acc,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes>& windows,
           const std::array<float, axes>& scale, const T& transform) {
    // Prepare the scratchpad with all the remaining memory requested by sub-kernels
    // TODO(michalz): Get rid of this run-time memory kind dispatch
    PreallocatedScratchpad sub_scratch;
    for (size_t i = 0; i < sub_scratch_sizes_.size(); i++) {
      auto sz = sub_scratch_sizes_[i];
      auto kind_id = static_cast<mm::memory_kind_id>(i);
      sub_scratch.allocs[i] =
          BumpAllocator(static_cast<char*>(ctx.scratchpad->Alloc(kind_id, sz, 64)), sz);
    }

    KernelContext sub_ctx = ctx;
    sub_ctx.scratchpad = &sub_scratch;

    // Clear the scratchpad for sub-kernels to reuse memory
    dy_kernel_.Run(sub_ctx, acc, in, windows[0], scale[0]);
    sub_scratch.Clear();
    dx_kernel_.Run(sub_ctx, out, in, windows[1], transform);
  }

  scratch_sizes_t sub_scratch_sizes_;
  DyKernel dy_kernel_;
  DxKernel dx_kernel_;
};

template <typename T, typename Intermediate, typename Out, typename In, typename W,
          bool has_channels>
struct LaplacianCpuBase<T, Intermediate, Out, In, W, 3, has_channels> {
  static constexpr int axes = 3;
  using DzKernel = PartialDeriv<Intermediate, In, W, axes, 0, has_channels,
                                conv_transform::TransScaleSat<Intermediate, W>>;
  using DyKernel = PartialDeriv<Intermediate, In, W, axes, 1, has_channels,
                                conv_transform::TransScaleAddOutSat<Intermediate, W>>;
  using DxKernel = PartialDeriv<Out, In, W, axes, 2, has_channels, T>;
  static constexpr int ndim = DzKernel::ndim;  // Dx and Dy ndim are the same

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<std::array<int, axes>, axes>& window_sizes) {
    KernelRequirements req;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in_shape));

    auto req_dz = dz_kernel_.Setup(ctx, in_shape, window_sizes[0]);
    auto req_dy = dy_kernel_.Setup(ctx, in_shape, window_sizes[1]);
    auto req_dx = dx_kernel_.Setup(ctx, in_shape, window_sizes[2]);

    // Calculate max scratch memory required by sub-kernels
    sub_scratch_sizes_ = MaxScratchSize(req_dx.scratch_sizes, req_dy.scratch_sizes);
    sub_scratch_sizes_ = MaxScratchSize(sub_scratch_sizes_, req_dz.scratch_sizes);
    req.scratch_sizes = sub_scratch_sizes_;

    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim>& out,
           const TensorView<StorageCPU, Intermediate, ndim>& acc,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes>& windows,
           const std::array<float, axes>& scale, const T& transform) {
    // Prepare the scratchpad with all the remaining memory requested by sub-kernels
    // TODO(michalz): Get rid of this run-time memory kind dispatch
    PreallocatedScratchpad sub_scratch;
    for (size_t i = 0; i < sub_scratch_sizes_.size(); i++) {
      auto sz = sub_scratch_sizes_[i];
      auto kind_id = static_cast<mm::memory_kind_id>(i);
      sub_scratch.allocs[i] =
          BumpAllocator(static_cast<char*>(ctx.scratchpad->Alloc(kind_id, sz, 64)), sz);
    }

    KernelContext sub_ctx = ctx;
    sub_ctx.scratchpad = &sub_scratch;

    // Clear the scratchpad for sub-kernels to reuse memory
    dz_kernel_.Run(sub_ctx, acc, in, windows[0], scale[0]);
    sub_scratch.Clear();
    dy_kernel_.Run(sub_ctx, acc, in, windows[1], scale[1]);
    sub_scratch.Clear();
    dx_kernel_.Run(sub_ctx, out, in, windows[2], transform);
  }

  scratch_sizes_t sub_scratch_sizes_;
  DzKernel dz_kernel_;
  DyKernel dy_kernel_;
  DxKernel dx_kernel_;
};


/**
 * @brief Laplacian kernel.
 * Provides separate specialization for 1 dimensional data. For other cases
 * there are two specializations depending on whether ``Intermediate`` is the same as ``Out``:
 * if not, the extra intermediate buffer is used to accumulate partial derivatives.
 *
 * For `axes` dimensional input data, there will be computed and summed `axes` partial derivatives,
 * one in each direction. The `i-th` partial derivative is computed by performing
 * a `axes`-dimensional separable convolution, where the `i-th` window is responsible for actual
 * derivative approximation and the remaining windows provide smoothing.
 * In total, there are ``axes * axes`` windows involved in computing laplacian. Thus,
 * the arguments that describe windows passed to the kernel methods are `axes x axes` dimensional
 * arrays of parameters, where ``[i][j]`` parameter refers to j-th window of i-th partial
 * derivative.
 *
 * @tparam Intermediate  Intermediate type used to accumulate partial derivatives.
 * @tparam Out           Desired output type. Conversion with clamping the output values
 *                       is performed if needed.
 * @tparam In            Input type.
 * @tparam W             Type of convolution window elements (see @ref SeparableConvolutionCpu).
 * @tparam axes          Number of spatial dimensions of the input data (and the number of partial
 *                       derivatives to compute).
 * @tparam has_channels  True iff the input data has ``axes + 1`` dimensions with the last
 *                       one treated as channels.
 * @tparam Dummy         Dummy type used with enable_if_t to disambiguate template specializations.
 */
template <typename Intermediate, typename Out, typename In, typename W, int axes, bool has_channels,
          typename Dummy = void>
struct LaplacianCpu;

template <typename Intermediate, typename Out, typename In, typename W, int axes, bool has_channels>
struct LaplacianCpu<Intermediate, Out, In, W, axes, has_channels,
                    std::enable_if_t<!std::is_same<Intermediate, Out>::value && axes != 1>>
    : LaplacianCpuBase<conv_transform::TransScaleAddBufferSat<Intermediate, Out, W>, Intermediate,
                       Out, In, W, axes, has_channels> {
  using Base = LaplacianCpuBase<conv_transform::TransScaleAddBufferSat<Intermediate, Out, W>,
                                Intermediate, Out, In, W, axes, has_channels>;
  using Base::ndim;

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<std::array<int, axes>, axes>& window_sizes) {
    auto req = Base::Setup(ctx, in_shape, window_sizes);
    ScratchpadEstimator se;
    se.add<mm::memory_kind::host, Intermediate>(volume(in_shape));
    req.scratch_sizes = AppendScratchSize(se.sizes, req.scratch_sizes);
    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim>& out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes>& windows,
           const std::array<float, axes>& scale) {
    auto* tmp = ctx.scratchpad->AllocateHost<Intermediate>(volume(in.shape));
    auto acc = TensorView<StorageCPU, Intermediate, ndim>(tmp, in.shape);
    Base::Run(ctx, out, acc, in, windows, scale, {tmp, scale[axes - 1]});
  }
};

template <typename Out, typename In, typename W, int axes, bool has_channels>
struct LaplacianCpu<Out, Out, In, W, axes, has_channels, std::enable_if_t<axes != 1>>
    : LaplacianCpuBase<conv_transform::TransScaleAddOutSat<Out, W>, Out, Out, In, W, axes,
                       has_channels> {
  using Base = LaplacianCpuBase<conv_transform::TransScaleAddOutSat<Out, W>, Out, Out, In, W, axes,
                                has_channels>;
  using Base::ndim;

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim>& out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes>& windows,
           const std::array<float, axes>& scale) {
    Base::Run(ctx, out, out, in, windows, scale, scale[axes - 1]);
  }
};

template <typename Intermediate, typename Out, typename In, typename W, bool has_channels>
struct LaplacianCpu<Intermediate, Out, In, W, 1, has_channels> {
  static constexpr int axes = 1;
  using ConvKernel = SeparableConvolutionCpu<Out, In, W, axes, has_channels,
                                             conv_transform::TransScaleSat<Out, W>>;
  static constexpr int ndim = ConvKernel::ndim;

  KernelRequirements Setup(KernelContext& ctx, const TensorShape<ndim>& in_shape,
                           const std::array<std::array<int, axes>, axes>& window_sizes) {
    return conv_.Setup(ctx, in_shape, window_sizes[0]);
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim>& out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes>& windows,
           const std::array<float, axes>& scale) {
    return conv_.Run(ctx, out, in, windows[0], scale[0]);
  }

  ConvKernel conv_;
};

}  // namespace laplacian

template <typename Out, typename In, typename W, int axes, bool has_channels>
using LaplacianCpu = laplacian::LaplacianCpu<decltype(std::declval<W>() * std::declval<In>()), Out,
                                             In, W, axes, has_channels>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_CPU_H_
