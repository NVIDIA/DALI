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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_cpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/convolution/gaussian_blur.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

using namespace gaussian_blur;  // NOLINT
using namespace convolution_utils;  // NOLINT

DALI_SCHEMA(GaussianBlur)
    .DocStr(R"code(Applies a Gaussian Blur to the input.

Gaussian blur is calculated by applying a convolution with a Gaussian kernel, which can be
parameterized with ``windows_size`` and ``sigma``.
If only the sigma is specified, the radius of the Gaussian kernel defaults to
``ceil(3 * sigma)``, so the kernel window size is ``2 * ceil(3 * sigma) + 1``.

If only the window size is provided, the sigma is calculated by using the following formula::

  radius = (window_size - 1) / 2
  sigma = (radius - 1) * 0.3 + 0.8

The sigma and kernel window size can be specified as one value for all data axes or a value
per data axis.

When specifying the sigma or window size per axis, the axes are provided same as layouts, from
outermost to innermost.

.. note::
  The channel ``C`` and frame ``F`` dimensions are not considered data axes. If channels are present,
  only channel-first or channel-last inputs are supported.

For example, with ``HWC`` input, you can provide ``sigma=1.0`` or ``sigma=(1.0, 2.0)`` because
there are two data axes, H and W.

The same input can be provided as per-sample tensors.
)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddOptionalArg<int>(kWindowSizeArgName, "The diameter of the kernel.",
                         std::vector<int>{0}, true)
    .AddOptionalArg<float>(kSigmaArgName, "Sigma value for the Gaussian Kernel.",
                           std::vector<float>{0.f}, true)
    .AddOptionalArg(
        "dtype", R"code(Output data type.

Supported type: `FLOAT`. If not set, the input type is used.)code",
        DALI_NO_TYPE);


namespace gaussian_blur {

// axes here is dimension of element processed by kernel - in case of sequence it's 1 less than the
// actual dim
template <typename Out, typename In, int axes, bool has_channels>
class GaussianBlurOpCpu : public OpImplBase<CPUBackend> {
 public:
  using Kernel = kernels::SeparableConvolutionCpu<Out, In, float, axes, has_channels>;
  static constexpr int ndim = Kernel::ndim;

  /**
   * @param spec  Pointer to a persistent OpSpec object,
   *              which is guaranteed to be alive for the entire lifetime of this object
   */
  explicit GaussianBlurOpCpu(const OpSpec* spec, const DimDesc& dim_desc)
      : spec_(*spec), dim_desc_(dim_desc) {}

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<CPUBackend>& ws) override {
    const auto& input = ws.template Input<CPUBackend>(0);
    int nsamples = input.num_samples();
    auto nthreads = ws.GetThreadPool().NumThreads();

    output_desc.resize(1);
    output_desc[0].type = type2id<Out>::value;
    output_desc[0].shape.resize(nsamples, input.shape().sample_dim());

    params_.resize(nsamples);
    windows_.resize(nsamples);

    kmgr_.template Resize<Kernel>(nsamples);

    for (int i = 0; i < nsamples; i++) {
      params_[i] = ObtainSampleParams<axes>(i, spec_, ws);
      windows_[i].PrepareWindows(params_[i]);
      // We take only last `ndim` siginificant dimensions to handle sequences as well
      auto elem_shape = input[i].shape().template last<ndim>();
      auto& req = kmgr_.Setup<Kernel>(i, ctx_, elem_shape, params_[i].window_sizes);
      // The shape of data stays untouched
      output_desc[0].shape.set_tensor_shape(i, input[i].shape());
    }
    return true;
  }

  void RunImpl(workspace_t<CPUBackend>& ws) override {
    const auto& input = ws.template Input<CPUBackend>(0);
    auto& output = ws.template Output<CPUBackend>(0);
    output.SetLayout(input.GetLayout());
    auto in_shape = input.shape();
    auto& thread_pool = ws.GetThreadPool();

    int nsamples = input.shape().num_samples();
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      const auto& shape = input[sample_idx].shape();
      auto elem_volume = volume(shape.begin() + dim_desc_.usable_axes_start, shape.end());

      int seq_elements = 1;
      int64_t stride = 0;
      if (dim_desc_.is_sequence()) {
        seq_elements = volume(shape.begin(), shape.begin() + dim_desc_.usable_axes_start);
        stride = elem_volume;
      }
      for (int elem_idx = 0; elem_idx < seq_elements; elem_idx++) {
        thread_pool.AddWork(
            [this, &input, &output, sample_idx, elem_idx, stride](int thread_id) {
              auto gaussian_windows = windows_[sample_idx].GetWindows();
              auto elem_shape = input[sample_idx].shape().template last<ndim>();
              auto in_view = TensorView<StorageCPU, const In, ndim>{
                  input[sample_idx].template data<In>() + stride * elem_idx, elem_shape};
              auto out_view = TensorView<StorageCPU, Out, ndim>{
                  output[sample_idx].template mutable_data<Out>() + stride * elem_idx, elem_shape};
              // I need a context for that particular run (or rather matching the thread &
              // scratchpad)
              auto ctx = ctx_;
              kmgr_.Run<Kernel>(sample_idx, ctx, out_view, in_view, gaussian_windows);
            }, elem_volume);
      }
    }
    thread_pool.RunAll();
  }

 private:
  const OpSpec &spec_;
  DimDesc dim_desc_;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;

  std::vector<GaussianBlurParams<axes>> params_;
  std::vector<GaussianWindows<axes>> windows_;
};


}  // namespace gaussian_blur

template <>
bool GaussianBlur<CPUBackend>::SetupImpl(std::vector<OutputDesc>& output_desc,
                                         const workspace_t<CPUBackend>& ws) {
  const auto& input = ws.template Input<CPUBackend>(0);
  auto layout = input.GetLayout();
  auto dim_desc = ParseAndValidateDim(input.shape().sample_dim(), layout);
  dtype_ = dtype_ != DALI_NO_TYPE ? dtype_ : input.type();
  DALI_ENFORCE(dtype_ == input.type() || dtype_ == DALI_FLOAT,
               "Output data type must be same as input, FLOAT or skipped (defaults to input type)");

  if (!impl_ || impl_in_dtype_ != input.type() || impl_dim_desc_ != dim_desc) {
    impl_in_dtype_ = input.type();
    impl_dim_desc_ = dim_desc;

    // clang-format off
    TYPE_SWITCH(input.type(), type2id, In, GAUSSIAN_BLUR_CPU_SUPPORTED_TYPES, (
      VALUE_SWITCH(dim_desc.usable_axes_count, Axes, GAUSSIAN_BLUR_SUPPORTED_AXES, (
        BOOL_SWITCH(dim_desc.is_channel_last(), HasChannels, (
          if (dtype_ == input.type()) {
            impl_ =
              std::make_unique<GaussianBlurOpCpu<In, In, Axes, HasChannels>>(&spec_, dim_desc);
          } else {
            impl_ =
              std::make_unique<GaussianBlurOpCpu<float, In, Axes, HasChannels>>(&spec_, dim_desc);
          }
        ));  // NOLINT
      ), DALI_FAIL("Axis count out of supported range."));  // NOLINT
    ), DALI_FAIL(make_string("Unsupported data type: ", input.type())));  // NOLINT
    // clang-format on
  }

  return impl_->SetupImpl(output_desc, ws);
}

template <>
void GaussianBlur<CPUBackend>::RunImpl(workspace_t<CPUBackend>& ws) {
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(GaussianBlur, GaussianBlur<CPUBackend>, CPU);

}  // namespace dali
