// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {
namespace kernels {

/**
 * @brief Apply convolution in all spatial axes, starting from the innermost to outermost.
 *        If channel axis is pressent, the convolution is not applied there.
 *        If it is marqed as sequence, the first data axis is considered as temporal axis
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
template <typename Out, typename In, typename W, int axes, bool has_channels = false, bool is_sequence = false>
struct SeparableConvolutionGpu;

// template <typename Out, typename In, typename W, bool has_channels>
// struct SeparableConvolutionGpu<Out, In, W, 1, has_channels, true> {
//   // static_assert(!is_sequence,
//   //               "1-axis data marked as sequence would contain only temporal axis thus no "
//   //               "convolution can be applied.");
//   static constexpr int axes = 1;
//   static constexpr int total_axes = 1; // this is to allow compilation
//   static constexpr int ndim = has_channels ? 2 : 1;

//   KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape,
//                            const std::array<TensorListShape<1>, axes>& window_sizes) {
//     DALI_FAIL("1-axis data marked as sequence would contain only temporal axis thus no "
//                 "convolution can be applied.");

//     return {};
//   }

//   void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
//            const TensorListView<StorageGPU, const In, ndim>& in,
//            const std::array<TensorListView<StorageCPU, const W, 1>, axes>& windows,
//            W scale = 1) {
//     DALI_FAIL("1-axis data marked as sequence would contain only temporal axis thus no "
//                 "convolution can be applied.");
//   }
// };

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
    req.AddInputSet(req_conv, false);

    return req;
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const std::array<TensorListView<StorageCPU, const W, 1>, axes>& windows,
           W scale = 1) {
    conv_.Run(ctx, out, in, windows[0], scale);
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
    se.add<Intermediate>(AllocType::GPU, in_shape.num_elements());
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(in_shape);

    auto req_inner = conv_innermost_.Setup(ctx, in_shape, window_sizes[1]);
    auto req_outer = conv_outermost_.Setup(ctx, in_shape, window_sizes[0]);

    req.AddInputSet(req_inner, false);
    req.AddInputSet(req_outer, false);

    return req;
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const std::array<TensorListView<StorageCPU, const W, 1>, axes>& windows,
           W scale = 1) {
    auto *tmp = ctx.scratchpad->Allocate<Intermediate>(AllocType::GPU, in.shape.num_elements());
    auto intermediate = TensorListView<StorageGPU, Intermediate, ndim>(tmp, in.shape);

    conv_innermost_.Run(ctx, intermediate, in, windows[1]);
    conv_outermost_.Run(ctx, out, intermediate, windows[0], scale);
  }

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

  KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape,
                           const std::array<TensorListShape<1>, axes>& window_sizes) {
    KernelRequirements req;

    ScratchpadEstimator se;
    // todo(klecki): ehh, not in place
    se.add<Intermediate>(AllocType::GPU, in_shape.num_elements() * 2);
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(in_shape);

    auto req_inner = conv_innermost_.Setup(ctx, in_shape, window_sizes[2]);
    auto req_middle = conv_middle_.Setup(ctx, in_shape, window_sizes[1]);
    auto req_outer = conv_outermost_.Setup(ctx, in_shape, window_sizes[0]);

    req.AddInputSet(req_inner, false);
    req.AddInputSet(req_middle, false);
    req.AddInputSet(req_outer, false);

    return req;
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const std::array<TensorListView<StorageCPU, const W, 1>, axes>& windows,
           W scale = 1) {
    auto* tmp = ctx.scratchpad->Allocate<Intermediate>(AllocType::GPU, in.shape.num_elements() * 2);
    auto intermediate_inner = TensorListView<StorageGPU, Intermediate, ndim>(tmp, in.shape);
    auto intermediate_outer = TensorListView<StorageGPU, Intermediate, ndim>(tmp + in.shape.num_elements(), in.shape);

    conv_innermost_.Run(ctx, intermediate_inner, in, windows[2]);
    // TODO(klecki): it probably doesn't work inplace - this will do for now,
    // but probably it's better to maybe do it in smaller batches/slices
    conv_middle_.Run(ctx, intermediate_outer, intermediate_inner, windows[1]);
    conv_outermost_.Run(ctx, out, intermediate_outer, windows[0], scale);
  }

  ConvolutionGpu<Intermediate, In, W, ndim, sequence_axes + 2, has_channels> conv_innermost_;
  ConvolutionGpu<Intermediate, Intermediate, W, ndim, sequence_axes + 1, has_channels> conv_middle_;
  ConvolutionGpu<Out, Intermediate, W, ndim, sequence_axes + 0, has_channels> conv_outermost_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_CONVOLUTION_GPU_H_
