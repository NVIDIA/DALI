// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/image/convolution/convolution_utils.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

namespace gaussian_blur {

using namespace convolution_utils;  // NOLINT

using op_impl_uptr = std::unique_ptr<OpImplBase<GPUBackend>>;

template <typename Out, typename In, int axes, bool has_channels>
class GaussianBlurOpGpu : public OpImplBase<GPUBackend> {
 public:
  using WindowType = float;
  using Kernel =
      kernels::SeparableConvolutionGpu<Out, In, WindowType, axes, has_channels, false>;
  static constexpr int ndim = Kernel::ndim;

  /**
   * @param spec  Pointer to a persistent OpSpec object,
   *              which is guaranteed to be alive for the entire lifetime of this object
   */
  explicit GaussianBlurOpGpu(const OpSpec* spec)
      : spec_(*spec) {}

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<GPUBackend>& ws) override {
    ctx_.gpu.stream = ws.stream();

    const auto& input = ws.template Input<GPUBackend>(0);
    auto processed_shape = input.shape();
    int nsamples = processed_shape.num_samples();
    constexpr int nthreads = 1;

    output_desc.resize(1);
    output_desc[0].type = type2id<Out>::value;
    output_desc[0].shape = input.shape();

    params_.resize(nsamples);
    windows_.resize(nsamples);
    for (auto &win_shape : window_shapes_) {
      win_shape.resize(nsamples);
    }
    for (auto &windows : windows_tl_) {
      windows.data.resize(nsamples);
      windows.shape.resize(nsamples);
    }

    kmgr_.template Resize<Kernel>(nsamples);

    for (int i = 0; i < nsamples; i++) {
      params_[i] = ObtainSampleParams<axes>(i, spec_, ws, get_frame_info<GPUBackend>(ws, 0));
      if (windows_[i].PrepareWindows(params_[i])) {
        for (int axis = 0; axis < axes; axis++) {
          window_shapes_[axis].set_tensor_shape(i, {params_[i].window_sizes[axis]});
          windows_tl_[axis].data[i] = windows_[i].GetWindows()[axis].data;
          windows_tl_[axis].shape.set_tensor_shape(i, windows_[i].GetWindows()[axis].shape);
        }
      }
    }
    auto& req = kmgr_.Setup<Kernel>(0, ctx_, processed_shape.to_static<ndim>(), window_shapes_);
    return true;
  }

  void RunImpl(workspace_t<GPUBackend>& ws) override {
    const auto& input = ws.template Input<GPUBackend>(0);
    auto& output = ws.template Output<GPUBackend>(0);
    output.SetLayout(input.GetLayout());

    auto in_view = view<const In, ndim>(input);
    auto out_view = view<Out, ndim>(output);
    kmgr_.Run<Kernel>(0, ctx_, out_view, in_view, windows_tl_);
  }

 private:
  const OpSpec &spec_;

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
std::unique_ptr<OpImplBase<GPUBackend>> GetGaussianBlurGpuImpl(const OpSpec* spec,
                                                               DimDesc dim_desc) {
  std::unique_ptr<OpImplBase<GPUBackend>> result;
  VALUE_SWITCH(dim_desc.axes, Axes, GAUSSIAN_BLUR_SUPPORTED_AXES, (
    BOOL_SWITCH(dim_desc.has_channels, HasChannels, (
      result.reset(
        new GaussianBlurOpGpu<Out, In, Axes, HasChannels>(spec));
    ));  // NOLINT
  ), DALI_FAIL("Axis count out of supported range."));  // NOLINT
  return result;
}

}  // namespace gaussian_blur
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_GPU_H_
