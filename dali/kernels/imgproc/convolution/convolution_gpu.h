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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_GPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_GPU_H_

#include "dali/core/boundary.h"
#include "dali/core/convert.h"
#include "dali/core/format.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/cutlass/device/gemm.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {
namespace kernels {

/**
 * @brief Apply convolution with 1-channel `window` in specified axis.
 *
 * Operation is done by using GEMM and generating the matrix from convolution window on the fly
 */
template <typename Out, typename In, typename W, int ndim, int axis, bool has_channels = true>
struct ConvolutionGpu {
  KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape,
                           const TensorListShape<1>& window_size) {
    KernelRequirements req;
    ScratchpadEstimator se;
    DALI_ENFORCE(
        in_shape.size() == window_size.size(),
        make_string("Provided input shape and window sizes do not mach in number of samples: ",
                    in_shape.size(), " vs ", window_size.size(), "."));
    int num_samples = in_shape.size();
    for (int i = 0; i < num_samples; i++) {
      int num_channels = has_channels ? in_shape[i][ndim - 1] : 1;
      DALI_ENFORCE(window_size[i][0] % 2 == 1,
                   make_string("Kernel window should have odd length, got: ", window_size,
                               " for sample ", i, "."));

      DALI_ENFORCE((window_size[i][0] / 2) * num_channels < kMaxRadiusSpan,
                   make_string("Kernel window too big for sample ", i, "."));
    }
    se.add<W>(AllocType::Host, num_samples * kWindowCopyBufferSize);
    se.add<W>(AllocType::GPU, num_samples * kWindowCopyBufferSize);
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(in_shape);
    return req;
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const TensorListView<StorageCPU, const W, 1>& window, W scale = 1) {
    int num_samples = in.size();
    auto* window_tmp_buffer_host =
        ctx.scratchpad->Allocate<W>(AllocType::Host, num_samples * kWindowCopyBufferSize);
    auto* window_tmp_buffer_gpu =
        ctx.scratchpad->Allocate<W>(AllocType::GPU, num_samples * kWindowCopyBufferSize);

    // Pad and align windows
    AlignWindows(window_tmp_buffer_host, window, in.shape);
    cudaMemcpyAsync(window_tmp_buffer_gpu, window_tmp_buffer_host,
                    sizeof(W) * num_samples * kWindowCopyBufferSize, cudaMemcpyHostToDevice,
                    ctx.gpu.stream);

    Arguments args;
    if (kInnerConv) {
      // Inner (or rather innermost) - repack arguments
      for (int i = 0; i < num_samples; i++) {
        cutlass::Array<int, 2> size;
        auto sample_shape = in.tensor_shape(i);
        int num_channels = has_channels ? sample_shape[ndim - 1] : 1;
        // height
        size[0] = volume(sample_shape.begin(), sample_shape.begin() + axis);
        // width
        size[1] = sample_shape[ndim - has_channels - 1];
        int row_stride = sample_shape[ndim - has_channels - 1] * num_channels;
        auto* window_gpu = window_tmp_buffer_gpu + i * kWindowCopyBufferSize;
        args.push_back(SampleArguments{
            size,                                              // Input matrix dimensions
            static_cast<int>(window.tensor_shape_span(i)[0]),  // Window sizes
            num_channels,                                      // channels count (innermost)
            {in.tensor_data(i), row_stride},                   // Tensor-ref for source matrix A
            window_gpu,                                        // Pointers to windows
            {out.tensor_data(i), row_stride},                  // Tensor-ref for source matrix C
            {out.tensor_data(i), row_stride},  // Tensor-ref for destination matrix D
            {scale, 0}                         // Scalars used in the Epilogue
        });
      }
    } else {
      // Outer or not-innermost - repack arguments
      for (int i = 0; i < num_samples; i++) {
        cutlass::Array<int, 2> size;
        auto sample_shape = in.tensor_shape(i);
        // height
        size[0] = sample_shape[axis];
        // width
        size[1] = volume(sample_shape.begin() + axis + 1, sample_shape.end());
        auto strides = GetStrides(sample_shape);
        int row_stride = strides[axis];
        int planes = volume(sample_shape.begin(), sample_shape.begin() + axis);
        int plane_stride = axis > 0 ? strides[axis - 1] : 0;
        auto* window_gpu = window_tmp_buffer_gpu + i * kWindowCopyBufferSize;
        args.push_back(SampleArguments{
            size,                                              // Input matrix dimensions
            static_cast<int>(window.tensor_shape_span(i)[0]),  // Window sizes
            1,                                 // channels don't matter for outer dimensions
            {in.tensor_data(i), row_stride},   // Tensor-ref for source matrix A
            window_gpu,                        // Pointers to windows
            {out.tensor_data(i), row_stride},  // Tensor-ref for source matrix C
            {out.tensor_data(i), row_stride},  // Tensor-ref for destination matrix D
            {scale, 0},                        // Scalars used in the Epilogue
            planes,
            plane_stride});
      }
    }
    // Construct and invoke the CUTLASS kernel
    CutlassConv gemm_operator;
    auto status = gemm_operator.can_implement(args);
    DALI_ENFORCE(status == cutlass::Status::kSuccess,
                 make_string("Operation not possible: ", cutlass::cutlassGetStatusString(status)));
    gemm_operator(args, ctx.gpu.stream);
  }

 private:
  static constexpr bool kInnerConv = axis == ndim - has_channels - 1;
  using RowMajor = cutlass::layout::RowMajor;
  // Basic SIMT kernel with no additional conversions
  using CutlassConv = typename cutlass::gemm::device::Conv<In, In,    // Data-type of Input matrix
                                                           RowMajor,  // Layout of Input matrix
                                                           W, W,      // Data-type of Conv window
                                                           Out,       // Data-type of Output matrix
                                                           RowMajor,  // Layout of Output matrix
                                                           kInnerConv>;

  static constexpr int kMaxRadiusSpan = CutlassConv::ConvWindowConfiguration::kMaxWindowRadiusSpan;
  static constexpr int kWindowCopyBufferSize =
      CutlassConv::ConvWindowConfiguration::kTotalAlignedSize;

  using Arguments = typename CutlassConv::Arguments;

  using SampleArguments = typename CutlassConv::SampleArguments;

  static_assert(0 <= axis && axis < (has_channels ? ndim - 1 : ndim),
                "Selected axis must be in [0, ndim) when there is no channel axis, or in [0, ndim "
                "- 1) for channel-last input");

  void AlignWindows(W* window_tmp_buffer_host, const TensorListView<StorageCPU, const W, 1>& window,
                    const TensorListShape<ndim>& in_shape) {
    for (int i = 0; i < window.num_samples(); i++) {
      using dst_win_t = typename CutlassConv::ConvWindowConfiguration::PaddedWindowBuffer<W>;
      int num_channels = has_channels ? in_shape[i][ndim - 1] : 1;
      auto window_src = make_span(window.tensor_data(i), window.tensor_shape_span(i)[0]);
      auto window_padded_dst = dst_win_t(window_tmp_buffer_host + i * kWindowCopyBufferSize);
      CutlassConv::ConvWindowConfiguration::prepare_window(window_padded_dst, window_src,
                                                           num_channels);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_GPU_H_
