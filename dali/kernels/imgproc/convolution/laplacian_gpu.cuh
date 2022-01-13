// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_GPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_GPU_H_

#include <memory>
#include <utility>
#include <vector>

#include "dali/core/dev_buffer.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/convolution_gpu.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_gpu.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/scratch.h"
#include "dali/pipeline/util/operator_impl_utils.h"


namespace dali {
namespace kernels {

namespace laplacian {

namespace detail {

struct CastSampleDesc {
  void* output;
  const void* input;
};

template <typename Out, typename In>
__global__ void BatchedCastKernel(const CastSampleDesc* samples,
                                  const kernels::BlockDesc<1>* blocks) {
  const auto& block = blocks[blockIdx.x];
  const auto& sample = samples[block.sample_idx];
  auto* out = reinterpret_cast<Out*>(sample.output);
  const auto* in = reinterpret_cast<const In*>(sample.input);
  for (int x = threadIdx.x + block.start.x; x < block.end.x; x += blockDim.x) {
    out[x] = ConvertSat<Out>(in[x]);
  }
}

}  // namespace detail

template <typename Out, typename In, typename W, int axes, int deriv_axis, bool has_channels,
          bool is_sequence>
struct PartialDerivGpu {
  using MultiDimConv = SeparableConvolutionGpu<Out, In, W, axes, has_channels, is_sequence>;
  static constexpr int ndim = MultiDimConv::ndim;
  static constexpr int sequence_axes = MultiDimConv::sequence_axes;
  using SingleDimConv = ConvolutionGpu<Out, In, W, ndim, sequence_axes + deriv_axis, has_channels>;

  KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape,
                           const std::array<TensorListShape<1>, axes>& window_sizes,
                           bool has_smoothing) {
    has_smoothing_ = has_smoothing;
    if (has_smoothing_) {
      if (!multi_dim_impl_) {
        single_dim_impl_.reset();
        multi_dim_impl_ = std::make_unique<MultiDimConv>();
      }
      return multi_dim_impl_->Setup(ctx, in_shape, window_sizes, true);
    }
    if (!single_dim_impl_) {
      multi_dim_impl_.reset();
      single_dim_impl_ = std::make_unique<SingleDimConv>();
    }
    return single_dim_impl_->Setup(ctx, in_shape, window_sizes[deriv_axis]);
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const std::array<TensorListView<StorageCPU, const W, 1>, axes>& windows,
           const ConvEpilogue& conv_epilogue) {
    if (has_smoothing_) {
      multi_dim_impl_->Run(ctx, out, in, windows, {}, conv_epilogue);
    } else {
      single_dim_impl_->Run(ctx, out, in, windows[deriv_axis], {}, conv_epilogue);
    }
  }

  bool has_smoothing_;
  std::unique_ptr<MultiDimConv> multi_dim_impl_;
  std::unique_ptr<SingleDimConv> single_dim_impl_;
};

/**
 * @brief Provides Laplacian specializations for 2 and 3 dimensional data.
 *  Run methods expect additional ``acc`` buffer that will be used to accumulate
 *  partial derivatives. If ``Intermediate`` and ``Out`` are the same type,
 *  the ``acc`` and ``out`` can be the same tensor.
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
      const std::array<std::array<TensorListShape<1>, axes>, axes>& window_sizes,
      bool has_smoothing) {
    KernelRequirements req;
    req.output_shapes.push_back(in_shape);

    auto req_dy = sobel_dy_.Setup(ctx, in_shape, window_sizes[0], has_smoothing);
    auto req_dx = sobel_dx_.Setup(ctx, in_shape, window_sizes[1], has_smoothing);

    // Calculate max scratch memory required by sub-kernels
    sub_scratch_sizes_ = MaxScratchSize(req_dx.scratch_sizes, req_dy.scratch_sizes);
    req.scratch_sizes = sub_scratch_sizes_;

    return req;
  }

  void Run(
      KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
      const TensorListView<StorageGPU, const In, ndim>& in,
      const std::array<std::array<TensorListView<StorageCPU, const W, 1>, axes>, axes>& windows,
      std::array<span<const float>, axes> scale) {
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
    sobel_dy_.Run(sub_ctx, out, in, windows[0], {scale[0]});
    sub_scratch.Clear();
    sobel_dx_.Run(sub_ctx, out, in, windows[1], {scale[1], 1.f});
  }

  scratch_sizes_t sub_scratch_sizes_;
  DyKernel sobel_dy_;
  DxKernel sobel_dx_;
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
      const std::array<std::array<TensorListShape<1>, axes>, axes>& window_sizes,
      bool has_smoothing) {
    KernelRequirements req;
    req.output_shapes.push_back(in_shape);

    auto req_dz = sobel_dz_.Setup(ctx, in_shape, window_sizes[0], has_smoothing);
    auto req_dy = sobel_dy_.Setup(ctx, in_shape, window_sizes[1], has_smoothing);
    auto req_dx = sobel_dx_.Setup(ctx, in_shape, window_sizes[2], has_smoothing);

    // Calculate max scratch memory required by sub-kernels
    sub_scratch_sizes_ = MaxScratchSize(req_dx.scratch_sizes, req_dy.scratch_sizes);
    sub_scratch_sizes_ = MaxScratchSize(sub_scratch_sizes_, req_dz.scratch_sizes);
    req.scratch_sizes = sub_scratch_sizes_;

    return req;
  }

  void Run(
      KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
      const TensorListView<StorageGPU, const In, ndim>& in,
      const std::array<std::array<TensorListView<StorageCPU, const W, 1>, axes>, axes>& windows,
      std::array<span<const float>, axes> scale) {
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
    sobel_dz_.Run(sub_ctx, out, in, windows[0], {scale[0]});
    sub_scratch.Clear();
    sobel_dy_.Run(sub_ctx, out, in, windows[1], {scale[1], 1.f});
    sub_scratch.Clear();
    sobel_dx_.Run(sub_ctx, out, in, windows[2], {scale[2], 1.f});
  }

  scratch_sizes_t sub_scratch_sizes_;
  DzKernel sobel_dz_;
  DyKernel sobel_dy_;
  DxKernel sobel_dx_;
};

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
      const std::array<std::array<TensorListShape<1>, axes>, axes>& window_sizes,
      bool has_smoothing) {
    (void)has_smoothing;
    return sobel_dx_.Setup(ctx, in_shape, window_sizes[0]);
  }

  void Run(
      KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
      const TensorListView<StorageGPU, const In, ndim>& in,
      const std::array<std::array<TensorListView<StorageCPU, const W, 1>, axes>, axes>& windows,
      std::array<span<const float>, axes> scale) {
    return sobel_dx_.Run(ctx, out, in, windows[0], {}, {scale[0]});
  }

  DxKernel sobel_dx_;
};

template <typename Intermediate, typename Out, typename In, typename W, int axes, bool has_channels,
          bool is_sequence>
struct LaplacianGpu<Intermediate, Out, In, W, axes, has_channels, is_sequence,
                    std::enable_if_t<!std::is_same<Intermediate, Out>::value && axes != 1>>
    : LaplacianGpuBase<Intermediate, In, W, axes, has_channels, is_sequence> {
  using Base = LaplacianGpuBase<Intermediate, In, W, axes, has_channels, is_sequence>;
  using Base::ndim;
  using GpuBlockSetup = kernels::BlockSetup<1, -1>;

  KernelRequirements Setup(
      KernelContext& ctx, const TensorListShape<ndim>& in_shape,
      const std::array<std::array<TensorListShape<1>, axes>, axes>& window_sizes,
      bool has_smoothing) {
    auto req = Base::Setup(ctx, in_shape, window_sizes, has_smoothing);
    ScratchpadEstimator se;
    se.add<mm::memory_kind::device, Intermediate>(in_shape.num_elements());
    req.scratch_sizes = AppendScratchSize(se.sizes, req.scratch_sizes);
    samples_.resize(in_shape.num_samples());
    std::array<std::pair<int, int>, 1> collapse_groups = {{{0, in_shape.sample_dim()}}};
    auto collapsed_shape = collapse_dims<1>(in_shape, collapse_groups);
    block_setup_.SetupBlocks(collapsed_shape, true);
    blocks_dev_.from_host(block_setup_.Blocks(), ctx.gpu.stream);
    return req;
  }

  void Run(
      KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
      const TensorListView<StorageGPU, const In, ndim>& in,
      const std::array<std::array<TensorListView<StorageCPU, const W, 1>, axes>, axes>& windows,
      std::array<span<const float>, axes> scale) {
    auto* tmp = ctx.scratchpad->AllocateGPU<Intermediate>(in.shape.num_elements());
    auto intermediate = TensorListView<StorageGPU, Intermediate, ndim>(tmp, in.shape);
    Base::Run(ctx, intermediate, in, windows, scale);
    int nsamples = intermediate.shape.num_samples();
    for (int sample_id = 0; sample_id < nsamples; sample_id++) {
      samples_[sample_id].output = out.tensor_data(sample_id);
      samples_[sample_id].input = intermediate.tensor_data(sample_id);
    }
    samples_dev_.from_host(samples_, ctx.gpu.stream);
    dim3 grid_dim = block_setup_.GridDim();
    dim3 block_dim = block_setup_.BlockDim();
    detail::BatchedCastKernel<Out, Intermediate>
        <<<grid_dim, block_dim, 0, ctx.gpu.stream>>>(samples_dev_.data(), blocks_dev_.data());
  }

  GpuBlockSetup block_setup_;
  std::vector<detail::CastSampleDesc> samples_;
  DeviceBuffer<GpuBlockSetup::BlockDesc> blocks_dev_;
  DeviceBuffer<detail::CastSampleDesc> samples_dev_;
};

}  // namespace laplacian

template <typename Out, typename In, typename W, int axes, bool has_channels, bool is_sequence>
using LaplacianGpu = laplacian::LaplacianGpu<decltype(std::declval<W>() * std::declval<In>()), Out,
                                             In, W, axes, has_channels, is_sequence>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_GPU_H_
