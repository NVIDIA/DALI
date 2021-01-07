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


#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_GPU_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_GPU_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_gpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/convolution/gaussian_blur.h"
#include "dali/operators/image/convolution/gaussian_blur_params.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

namespace gaussian_blur {

using op_impl_uptr = std::unique_ptr<OpImplBase<GPUBackend>>;

template <typename Out, typename In, int axes, bool has_channels, bool is_sequence>
class GaussianBlurOpGpu : public OpImplBase<GPUBackend> {
 public:
  using WindowType = float;
  using Kernel =
      kernels::SeparableConvolutionGpu<Out, In, WindowType, axes, has_channels, is_sequence>;
  static constexpr int ndim = Kernel::ndim;

  explicit GaussianBlurOpGpu(const OpSpec& spec, const DimDesc& dim_desc)
      : spec_(spec), dim_desc_(dim_desc) {}

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<GPUBackend>& ws) override {
    ctx_.gpu.stream = ws.stream();

    const auto& input = ws.template InputRef<GPUBackend>(0);
    auto processed_shape = input.shape();
    int nsamples = processed_shape.num_samples();
    // If we are sequence-like, make sure that all sequence elements are compressed to first dim
    if (is_sequence) {
      processed_shape = collapse_dims(processed_shape, {{0, dim_desc_.usable_axes_start}});
    }
    constexpr int nthreads = 1;

    output_desc.resize(1);
    output_desc[0].type = TypeInfo::Create<Out>();
    // The shape of data stays untouched
    output_desc[0].shape = input.shape();

    params_.resize(nsamples);
    windows_.resize(nsamples);
    for (auto &win_shape : window_shapes_) {
      win_shape.resize(nsamples);
    }

    kmgr_.template Resize<Kernel>(nthreads, nsamples);

    for (int i = 0; i < nsamples; i++) {
      params_[i] = ObtainSampleParams<axes>(i, spec_, ws);
      windows_[i].PrepareWindows(params_[i]);
    }
    RepackAsTL<axes>(window_shapes_, params_);
    RepackAsTL<axes>(windows_tl_, windows_);
    auto& req = kmgr_.Setup<Kernel>(0, ctx_, processed_shape.to_static<ndim>(), window_shapes_);
    return true;
  }

  void RunImpl(workspace_t<GPUBackend>& ws) override {
    const auto& input = ws.template InputRef<GPUBackend>(0);
    auto& output = ws.template OutputRef<GPUBackend>(0);
    output.SetLayout(input.GetLayout());

    auto processed_shape = input.shape();
    int nsamples = processed_shape.num_samples();
    // If we are sequence-like, make sure that all sequence elements are compressed to first dim
    if (is_sequence) {
      processed_shape = collapse_dims(processed_shape, {{0, dim_desc_.usable_axes_start}});
    }

    auto static_shape = processed_shape.to_static<ndim>();

    auto in_view_dyn = view<const In>(input);
    auto out_view_dyn = view<Out>(output);

    // TODO(klecki): Just create it from the move(in_view_dyn.data), processed_shape
    auto in_view = reshape<ndim>(in_view_dyn, static_shape);
    auto out_view = reshape<ndim>(out_view_dyn, static_shape);

    kmgr_.Run<Kernel>(0, 0, ctx_, out_view, in_view, windows_tl_);
  }

 private:
  OpSpec spec_;
  DimDesc dim_desc_;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;

  std::vector<GaussianBlurParams<axes>> params_;
  std::vector<GaussianWindows<axes>> windows_;
  std::array<TensorListShape<1>, axes> window_shapes_;
  std::array<TensorListView<StorageCPU, const float, 1>, axes> windows_tl_;
};

/**
 * @brief Obtain an instance of GaussianBlurGpuImpl for given `Out` and `In` types
 * and dimensionality provided by runtime DimDesc.
 *
 * This function is explicitly instantiated in gaussian_blur_impl_[type].cu files
 * to allow for parallel compilation of underlying kernels.
 */
template <typename Out, typename In>
std::unique_ptr<OpImplBase<GPUBackend>> GetGaussianBlurGpuImpl(const OpSpec& spec,
                                                               DimDesc dim_desc) {
  std::unique_ptr<OpImplBase<GPUBackend>> result;
  VALUE_SWITCH(dim_desc.usable_axes_count, AXES, GAUSSIAN_BLUR_SUPPORTED_AXES, (
    VALUE_SWITCH(static_cast<int>(dim_desc.has_channels), HAS_CHANNELS, (0, 1), (
      VALUE_SWITCH(static_cast<int>(dim_desc.is_sequence), IS_SEQUENCE, (0, 1), (
        constexpr bool has_ch = HAS_CHANNELS;
        constexpr bool is_seq = IS_SEQUENCE;
          result.reset(new GaussianBlurOpGpu<Out, In, AXES, has_ch, is_seq>(spec, dim_desc));
      ), ());  // NOLINT, no other possible conversion
    ), ());  // NOLINT, no other possible conversion
  ), DALI_FAIL("Axis count out of supported range."));  // NOLINT
  return result;
}

}  // namespace gaussian_blur
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_GPU_H_
