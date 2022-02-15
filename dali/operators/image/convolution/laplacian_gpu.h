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

#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_GPU_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_GPU_H_

#include <memory>
#include <vector>

#include "dali/core/span.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/laplacian_gpu.cuh"
#include "dali/kernels/imgproc/convolution/laplacian_windows.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/convolution/laplacian.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"


namespace dali {

using namespace convolution_utils;  // NOLINT

namespace laplacian {

using op_impl_uptr = std::unique_ptr<OpImplBase<GPUBackend>>;

template <typename Out, typename In, int axes, bool has_channels, bool is_sequence>
class LaplacianOpGpu : public OpImplBase<GPUBackend> {
 public:
  using Kernel = kernels::LaplacianGpu<Out, In, float, axes, has_channels, is_sequence>;
  static constexpr int ndim = Kernel::ndim;

  /**
   * @param spec  Pointer to a persistent OpSpec object,
   *              which is guaranteed to be alive for the entire lifetime of this object
   */
  explicit LaplacianOpGpu(const OpSpec* spec, const DimDesc& dim_desc)
      : spec_{*spec}, args{*spec}, dim_desc_{dim_desc}, lap_windows_{maxWindowSize} {
    kmgr_.Resize<Kernel>(1);
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<GPUBackend>& ws) override {
    ctx_.gpu.stream = ws.stream();

    const auto& input = ws.template Input<GPUBackend>(0);
    auto processed_shape = input.shape();
    int nsamples = processed_shape.num_samples();
    // If we are sequence-like, make sure that all sequence elements are compressed to first dim
    if (is_sequence) {
      processed_shape = collapse_dims(processed_shape, {{0, dim_desc_.usable_axes_start}});
    }

    output_desc.resize(1);
    output_desc[0].type = type2id<Out>::value;
    // The shape of data stays untouched
    output_desc[0].shape = input.shape();

    args.ObtainLaplacianArgs(spec_, ws, nsamples);

    for (int i = 0; i < axes; i++) {
      for (int j = 0; j < axes; j++) {
        window_sizes_[i][j].resize(nsamples);
        windows_[i][j].resize(nsamples);
      }
      scales_[i].resize(nsamples);
      scale_spans_[i] = make_span(scales_[i]);
    }

    std::array<bool, axes> has_smoothing = uniform_array<axes>(false);
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      const auto& window_sizes = args.GetWindowSizes(sample_idx);
      const auto& scales = args.GetScales(sample_idx);
      for (int i = 0; i < axes; i++) {
        for (int j = 0; j < axes; j++) {
          if (i != j && window_sizes[i][j] != 1) {
            has_smoothing[i] = true;
          }
          window_sizes_[i][j].set_tensor_shape(sample_idx, {window_sizes[i][j]});
          auto window_size = window_sizes[i][j];
          const auto& window = i == j ? lap_windows_.GetDerivWindow(window_size) :
                                        lap_windows_.GetSmoothingWindow(window_size);
          windows_[i][j].data[sample_idx] = window.data;
          windows_[i][j].shape.set_tensor_shape(sample_idx, window.shape);
        }
        scales_[i][sample_idx] = scales[i];
      }
    }
    for (int i = 0; i < axes; i++) {
      if (!has_smoothing[i]) {
        for (int j = 0; j < axes; j++) {
          if (i != j) {
            window_sizes_[i][j].resize(0);
            windows_[i][j].resize(0);
          }
        }
      }
    }

    auto& req = kmgr_.Setup<Kernel>(0, ctx_, processed_shape.to_static<ndim>(), window_sizes_);
    return true;
  }

  void RunImpl(workspace_t<GPUBackend>& ws) override {
    const auto& input = ws.template Input<GPUBackend>(0);
    auto& output = ws.template Output<GPUBackend>(0);
    output.SetLayout(input.GetLayout());

    auto processed_shape = input.shape();
    // If we are sequence-like, make sure that all sequence elements are compressed to first dim
    if (is_sequence) {
      processed_shape = collapse_dims(processed_shape, {{0, dim_desc_.usable_axes_start}});
    }

    auto static_shape = processed_shape.to_static<ndim>();
    auto in_view_dyn = view<const In>(input);
    auto out_view_dyn = view<Out>(output);
    auto in_view = reshape<ndim>(in_view_dyn, static_shape);
    auto out_view = reshape<ndim>(out_view_dyn, static_shape);

    kmgr_.Run<Kernel>(0, ctx_, out_view, in_view, windows_, scale_spans_);
  }

 private:
  const OpSpec& spec_;
  LaplacianArgs<axes> args;
  DimDesc dim_desc_;
  kernels::LaplacianWindows<float> lap_windows_;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;

  std::array<std::array<TensorListShape<1>, axes>, axes> window_sizes_;
  std::array<std::vector<float>, axes> scales_;
  std::array<span<const float>, axes> scale_spans_;
  std::array<std::array<TensorListView<StorageCPU, const float, 1>, axes>, axes> windows_;
};

/**
 * @brief Obtain an instance of LaplacianOpGpu for given `Out` and `In` types
 * and dimensionality provided by runtime DimDesc.
 *
 * This function is explicitly instantiated in laplacian_impl_[type].cu files
 * to allow for parallel compilation of underlying kernels.
 */
template <typename Out, typename In>
op_impl_uptr GetLaplacianGpuImpl(const OpSpec* spec, const DimDesc& dim_desc) {
  op_impl_uptr result;
  VALUE_SWITCH(dim_desc.usable_axes_count, Axes, LAPLACIAN_SUPPORTED_AXES, (
    BOOL_SWITCH(dim_desc.is_channel_last(), HasChannels, (
      BOOL_SWITCH(dim_desc.is_sequence(), IsSeq, (
        using LaplacianImpl = LaplacianOpGpu<Out, In, Axes, HasChannels, IsSeq>;
        result.reset(new LaplacianImpl(spec, dim_desc))
      ));  // NOLINT
    ));  // NOLINT
  ), DALI_FAIL("Axis count out of supported range."));  // NOLINT
  return result;
}

}  // namespace laplacian

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_GPU_H_
