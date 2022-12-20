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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_CUH_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_CUH_

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "dali/core/cuda_rt_utils.h"
#include "dali/core/geom/vec.h"
#include "dali/core/span.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/filter_gpu_impl.cuh"
#include "dali/kernels/imgproc/roi.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

/**
 * @brief Computes the 2D or 3D (``axes_``) convolution of the input (``In``) and the
 * filter (``W``).
 *
 * The input must have the same number of spatial dims as the filter, but
 * can have extra sequence dim at the beginning (``has_sequence_dim``) and channels
 * dim at the end (``has_channel_dim``). If the ``enable_roi_`` is false, the outputs and inputs
 * must have the same size, otherwise they can differ and by passing arbitrary
 * ``anchors`` to ``Run``, the ROI can be specified.
 *
 * For details on the way the convolution is computed, see the docs of the actual cuda kernel.
 */
template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim,
          int axes_, bool enable_roi_>
struct FilterGpu {
  /* It computes a correlation of the input and the filter.
  Flip filter in both dimensions for a convolution. */

  static constexpr int axes = axes_;
  static constexpr bool enable_roi = enable_roi_;
  static constexpr int ndim =
      static_cast<int>(has_sequence_dim) + axes + static_cast<int>(has_channel_dim);
  static constexpr int sequence_dim = has_sequence_dim ? 0 : -1;
  static constexpr int channels_dim = has_channel_dim ? ndim - 1 : -1;
  static_assert(axes == 2 || axes == 3);
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());
  using StaticConfigT = filter::StaticConfig<axes>;
  using BlockSetupProviderT = filter::AdaptiveBlock<StaticConfigT>;
  using BlockSetupT = typename BlockSetupProviderT::BlockSetup;
  using StaticBlockFactoryT = filter::StaticBlock<StaticConfigT>;
  using StaticBlockT = typename StaticBlockFactoryT::BlockSetup;
  using GridSetupT = filter::GridSetup<StaticConfigT>;
  using InShapeDescT = filter::InShapeDesc<axes>;
  using OutShapeDescT = filter::OutShapeDesc<axes>;
  using WorkspaceDescT = filter::WorkspaceDesc<axes>;
  using SampleDescT = filter::SampleDesc<Out, In, W, Intermediate, axes>;

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim>& out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const TensorListView<StorageGPU, const W, axes>& filters,
           const span<const ivec<axes>> anchors,
           boundary::BoundaryType border_type,
           const TensorListView<StorageGPU, const In, 0>& fill_values = {}) {
    auto num_samples = in.shape.num_samples();
    assert(out.num_samples() == num_samples && filters.num_samples() == num_samples &&
           anchors.size() == num_samples);
    assert(fill_values.num_samples() == num_samples || fill_values.num_samples() == 0);

    SetupSampleDescs(out, in, filters, anchors);
    SampleDescT* samples_desc_dev;
    std::tie(samples_desc_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, samples_desc_);
    auto grid_setup = PrepareGridSetup(make_cspan(samples_desc_));
    WithBlockSetupProvider(ctx, [&](auto&& block_setup_provider) {
      WithInLoaderProvider(ctx, border_type, fill_values, [&](auto&& in_loader_provider) {
        WithOutShapeProvider([&](auto&& out_shape_provider) {
          WithConvFactory([&](auto&& conv_factory) {
            filter::filter<<<grid_setup.kernel_setup(), StaticConfigT::threadblock_size,
                             required_shm_size_, ctx.gpu.stream>>>(
                samples_desc_dev, block_setup_provider, in_loader_provider, grid_setup,
                out_shape_provider, conv_factory);
            CUDA_CALL(cudaGetLastError());
          });
        });
      });
    });
  }

 protected:
  template <typename T>
  static vec<axes, T> ShapeAsVec(const TensorShape<ndim>& shape) {
    return shape2vec<axes, T>(skip_dim<sequence_dim>(skip_dim<channels_dim>(shape)));
  }

  template <typename Inshapes, typename FilterShapes>
  void ValidateNumericLimits(const Inshapes& in_shapes, const FilterShapes& filter_shapes) {
    const i64vec<axes> max_grid_logical_extents =
        static_cast<i64vec<axes>>(StaticConfigT::max_grid_extents() * StaticConfigT::lanes);
    // so that we can safely use grid extents as a stride in a for loop
    const i64vec<axes> max_sample_extents =
        std::numeric_limits<int>::max() - max_grid_logical_extents + 1;
    for (int sample_idx = 0; sample_idx < in_shapes.num_samples(); sample_idx++) {
      const auto& in_shape = in_shapes[sample_idx];
      int64_t num_frames = has_sequence_dim ? in_shape[sequence_dim] : 1;
      if (num_frames > std::numeric_limits<int>::max()) {
        throw std::range_error(make_string(
            "The number of frames for sample of idx ", sample_idx, " exceeds the limit of ",
            std::numeric_limits<int>::max(), ". Got ", num_frames, " frames."));
      }
      int64_t num_channels = has_channel_dim ? in_shape[channels_dim] : 1;
      i64vec<axes> in_extents = ShapeAsVec<int64_t>(in_shape);
      in_extents[0] *= num_channels;
      for (int dim = 0; dim < axes; dim++) {
        const std::array<std::string, 3> extent_names = {
            "combined width and the number of channels", "height", "depth"};
        if (in_extents[dim] > max_sample_extents[dim]) {
          throw std::range_error(
              make_string("The ", extent_names[dim], " of sample at index ", sample_idx, " is ",
                          in_extents[dim], ", which exceeds the supported limit of ",
                          max_sample_extents[dim], ". The sample's shape is: ", in_shape, "."));
        }
      }
      const auto& filter_shape = filter_shapes[sample_idx];
      if (volume(filter_shape) > std::numeric_limits<int>::max()) {
        throw std::range_error(make_string(
            "Volume of filter for sample of idx ", sample_idx, " is ", volume(filter_shape),
            ", which exceeds the limit of ", std::numeric_limits<int>::max(),
            ". The filter's shape is ", filter_shape, "."));
      }
    }
  }

  void SetupSampleDescs(const TensorListView<StorageGPU, Out, ndim>& out,
                        const TensorListView<StorageGPU, const In, ndim>& in,
                        const TensorListView<StorageGPU, const W, axes>& filters,
                        const span<const ivec<axes>> anchors) {
    const auto& out_shapes = out.shape;
    const auto& in_shapes = in.shape;
    const auto& filter_shapes = filters.shape;
    int num_samples = in_shapes.num_samples();
    ValidateNumericLimits(in_shapes, filter_shapes);
    samples_desc_.clear();
    samples_desc_.reserve(num_samples);
    block_setups_.clear();
    block_setups_.reserve(num_samples);
    all_fit_in_shm_workspace_ = true;
    required_shm_size_ = 0;
    const int shared_mem_limit = GetSharedMemPerBlock();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& in_shape = in_shapes[sample_idx];
      int num_frames = has_sequence_dim ? in_shape[sequence_dim] : 1;
      int num_channels = has_channel_dim ? in_shape[channels_dim] : 1;
      ivec<axes> in_extents = ShapeAsVec<int>(in_shape);
      auto filter_extents = shape2vec(filter_shapes[sample_idx]);
      BlockSetupT block_setup{in_extents, num_channels};
      int lane_axis = GetLanesAxis(filter_extents);
      ivec<axes> logical_block_extents = block_setup.block_dim();
      logical_block_extents[lane_axis] *= StaticConfigT::lanes;
      WorkspaceDescT workspace_desc;
      int required_workspace;
      SetupWorkspaceDesc(workspace_desc, required_workspace, filter_extents, logical_block_extents,
                         num_channels, shared_mem_limit);
      InShapeDescT shape_desc = PrepareSampleShapeDesc(
          in_extents, filter_extents, anchors[sample_idx], num_frames, num_channels);
      ivec<axes> out_extents = ShapeAsVec<int>(out_shapes[sample_idx]);
      OutShapeDescT out_shape_desc = PrepareOutputShapeDesc(out_extents, num_channels);
      required_shm_size_ = std::max(required_shm_size_, required_workspace);
      all_fit_in_shm_workspace_ &= required_workspace > 0;
      block_setups_.push_back(block_setup);
      samples_desc_.push_back({out.tensor_data(sample_idx), in.tensor_data(sample_idx),
                               filters.tensor_data(sample_idx), shape_desc, out_shape_desc,
                               workspace_desc, logical_block_extents, lane_axis});
    }
  }

  InShapeDescT PrepareSampleShapeDesc(ivec<axes> in_extents, ivec<axes> filter_extents,
                                      ivec<axes> anchor_shift, int num_frames, int num_channels) {
    int width = in_extents.x;
    // meld the innermost dimension and channels to have non-strided innermost extent
    // kernel does not need to know the excat channel position most of the time
    // (only for OOB handling)
    in_extents.x *= num_channels;
    i64vec<axes> in_strides;
    int64_t frame_stride = CalcStrides<false>(in_strides, in_extents);
    ivec<axes> filter_strides;
    CalcStrides<false>(filter_strides, filter_extents);
    anchor_shift.x *= num_channels;
    return {frame_stride, in_strides,     num_frames,
            width,        num_channels,   filter_extents.x * num_channels,
            in_extents,   filter_extents, filter_strides,
            anchor_shift};
  }

  void SetupWorkspaceDesc(WorkspaceDescT& workspace_desc, int& required_workspace,
                          ivec<axes> filter_extents, ivec<axes> logical_block_extents,
                          int num_channels, int shared_mem_limit) {
    i64vec<axes> in_workspace_extents = static_cast<i64vec<axes>>(filter_extents) - 1;
    in_workspace_extents.x *= num_channels;
    in_workspace_extents += logical_block_extents;
    int64_t in_workspace_num_elements = volume(in_workspace_extents);
    int64_t idx_workspace_size =
        std::accumulate(in_workspace_extents.begin() + 1, in_workspace_extents.end(), 0);
    int64_t in_workspace_offset = align_up(idx_workspace_size * sizeof(int), sizeof(In));
    int64_t workspace_size = in_workspace_offset + in_workspace_num_elements * sizeof(In);
    // When iterating over the innermost filter extent, the logical block strides over the input
    // with the stride equal to the number of channels. If the stride is lower than the block size,
    // then diffrent positions of the block over the input overlap. The shm kernel relies on that
    // when transferring input to shared memory workspace.
    // For `num_channels > logical_block_extents.x`, it would transfer the gaps between different
    // block placements unnecessarily.
    if (num_channels > logical_block_extents.x || workspace_size > shared_mem_limit) {
      in_workspace_extents = workspace_size = in_workspace_offset = 0;
    }
    required_workspace = workspace_size;
    workspace_desc.in_offset = in_workspace_offset;
    workspace_desc.in_extents = in_workspace_extents;
    CalcStrides<false>(workspace_desc.in_strides, workspace_desc.in_extents);
  }

  OutShapeDescT PrepareOutputShapeDesc(ivec<axes> out_extents, int num_channels) {
    out_extents.x *= num_channels;
    i64vec<axes> out_strides;
    int64_t out_frame_stride = CalcStrides<false>(out_strides, out_extents);
    return {out_frame_stride, out_strides, out_extents};
  }

  GridSetupT PrepareGridSetup(span<const SampleDescT> sample_descs) {
    ivec<axes> num_blocks = 0;
    for (const auto& sample_desc : sample_descs) {
      auto sample_num_blocks =
          div_ceil(sample_desc.out_shape.extents, sample_desc.logical_block_extents);
      num_blocks = max(num_blocks, sample_num_blocks);
    }
    num_blocks = min(num_blocks, StaticConfigT::max_grid_extents());
    int num_samples = sample_descs.size();
    return {num_blocks, num_samples};
  }

  int GetLanesAxis(const ivec2 filter_extents) {
    (void)filter_extents;
    return 1;
  }

  int GetLanesAxis(const ivec3 filter_extents) {
    return filter_extents.y > filter_extents.z ? 1 : 2;
  }

  /**
   * @brief Selects block setup provider for kernel launch.
   *
   * Picks static block provider if possible (which has less arithmetic overhead) or adaptive,
   * per-sample provider if needed (i.e. when samples have degenerated extents such
   * as width = 1, which would result in very slow run with static block).
   */
  template <typename KernelLauncher>
  void WithBlockSetupProvider(KernelContext& ctx, KernelLauncher&& launch_kernel) {
    if (std::all_of(block_setups_.begin(), block_setups_.end(), [](const auto& block_setup) {
          return block_setup.block_dim() == StaticBlockT{}.block_dim();
        })) {
      launch_kernel(StaticBlockFactoryT{});
    } else {
      BlockSetupT* block_setups_dev;
      std::tie(block_setups_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, block_setups_);
      BlockSetupProviderT block_setup{block_setups_dev};
      launch_kernel(std::move(block_setup));
    }
  }

  /**
   * @brief Selected input loader provider for kernel launch that handles OOB accesess according to
   * selected border mode.
   */
  template <typename KernelLauncher>
  void WithInLoaderProvider(KernelContext& ctx, boundary::BoundaryType border_type,
                            const TensorListView<StorageGPU, const In, 0>& fill_values,
                            KernelLauncher&& launch_kernel) {
    using namespace boundary;  // NOLINT(build/namespaces)
    VALUE_SWITCH(border_type, BT, (
      BoundaryType::REFLECT_101, BoundaryType::REFLECT_1001,
      BoundaryType::CLAMP, BoundaryType::WRAP), (
        RunKernelBorderRemap<BT>(ctx, std::move(launch_kernel));
      ), (  // NOLINT
        if (border_type == BoundaryType::CONSTANT) {
          RunKernelBorderConstant(ctx, fill_values, std::move(launch_kernel));
        } else {
          DALI_FAIL(
            make_string("Unsupported border type was specified: ", to_string(border_type), "."));
        }
      ));  // NOLINT
  }

  template <boundary::BoundaryType border, typename KernelLauncher>
  void RunKernelBorderRemap(KernelContext& ctx, KernelLauncher&& launch_kernel) {
    using Loader = filter::InLoaderBorderRemap<In, axes, border>;
    filter::InLoaderProvider<Loader> loader_factory{};
    launch_kernel(std::move(loader_factory));
  }

  template <typename KernelLauncher>
  void RunKernelBorderConstant(KernelContext& ctx,
                               const TensorListView<StorageGPU, const In, 0>& fill_values,
                               KernelLauncher&& launch_kernel) {
    int num_samples = samples_desc_.size();
    if (fill_values.num_samples() != num_samples) {
      filter::InLoaderProvider<filter::InLoaderPad<In, axes>> loader_factory{nullptr};
      launch_kernel(std::move(loader_factory));
    } else {
      fill_values_.resize(num_samples);
      for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        fill_values_[sample_idx] = fill_values[sample_idx].data;
      }
      const In** fill_values_dev;
      std::tie(fill_values_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, fill_values_);
      filter::InLoaderProvider<filter::InLoaderPad<In, axes>> loader_factory{fill_values_dev};
      launch_kernel(std::move(loader_factory));
    }
  }

  /**
   * @brief If the ``enable_roi`` is set to false, the input and outpus are expected do have equal
   * shapes. Knowing that helps to spare a few registers when running the kernel.
   */
  template <typename KernelLauncher>
  void WithOutShapeProvider(KernelLauncher&& launch_kernel) {
    if (enable_roi) {
      launch_kernel(filter::OutShapeProviderROI<SampleDescT>{});
    } else {
      assert(std::all_of(samples_desc_.begin(), samples_desc_.end(), [](const auto& sample_desc) {
        return sample_desc.out_shape.extents == sample_desc.in_shape.in_extents;
      }));
      launch_kernel(filter::OutShapeProviderSame<SampleDescT>{});
    }
  }

  /**
   * @brief Runs the kernel with convolution that utilizes cuda shm if all samples
   * can fit there, otherwise chooses slower but more generic direct conv.
   */
  template <typename KernelLauncher>
  void WithConvFactory(KernelLauncher&& launch_kernel) {
    if (all_fit_in_shm_workspace_) {
      launch_kernel(filter::ShmConvFactory{});
    } else {
      launch_kernel(filter::DirectConvFactory{});
    }
  }

  std::vector<SampleDescT> samples_desc_;
  std::vector<const In*> fill_values_;
  std::vector<BlockSetupT> block_setups_;
  bool all_fit_in_shm_workspace_;
  int required_shm_size_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_CUH_
