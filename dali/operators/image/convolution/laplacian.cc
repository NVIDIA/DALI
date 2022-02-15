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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/laplacian_cpu.h"
#include "dali/kernels/imgproc/convolution/laplacian_windows.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/convolution/laplacian.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

using namespace convolution_utils;  // NOLINT

DALI_SCHEMA(Laplacian)
    .DocStr(R"code(Computes the Laplacian of an input.

The Laplacian is calculated as the sum of second order partial derivatives with respect to each
spatial dimension. Each partial derivative is approximated with a separable convolution,
that uses a derivative window in the direction of the partial derivative and smoothing windows
in the remaining axes.

By default, each partial derivative is approximated by convolving along all spatial axes: the axis
in partial derivative direction uses derivative window of ``window_size`` and the remaining
axes are convolved with smoothing windows of the same size. If ``smoothing_size`` is specified,
the smoothing windows applied to a given axis can have different size than the derivative window.
Specifying ``smoothing_size = 1`` implies no smoothing in axes perpendicular
to the derivative direction.

Both ``window_size`` and ``smoothing_size`` can be specified as a single value or per axis.
For example, for volumetric input, if ``window_size=[dz, dy, dx]``
and ``smoothing_size=[sz, sy, sx]`` are specified, the following windows will be used:

  * for partial derivative in ``z`` direction: derivative windows of size ``dz`` along ``z`` axis,
    and smoothing windows of size ``sy`` and ``sx`` along `y` and `x` respectively.
  * for partial derivative in ``y`` direction: derivative windows of size ``dy`` along ``y`` axis,
    and smoothing windows of size ``sz`` and ``sx`` along `z` and `x` respectively.
  * for partial derivative in ``x`` direction: derivative windows of size ``dx`` along ``x`` axis,
    and smoothing windows of size ``sz`` and ``sy`` along `z` and `y` respectively.

Window sizes and smoothing sizes must be odd. The size of a derivative window must be at least 3.
Smoothing window can be of size 1, which implies no smoothing along corresponding axis.

To normalize partial derivatives, ``normalized_kernel=True`` can be used. Each partial derivative
is scaled by ``2^(-s + n + 2)``, where ``s`` is the sum of the window sizes used to calculate
a given partial derivative (including the smoothing windows) and ``n`` is the number of data
dimensions/axes. Alternatively, you can specify ``scale`` argument to customize scaling factors.
Scale can be either a single value or ``n`` values, one for every partial derivative.

Operator uses 32-bit floats as an intermediate type.

.. note::
  The channel ``C`` and frame ``F`` dimensions are not considered data axes. If channels are present,
  only channel-first or channel-last inputs are supported.

)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddOptionalArg<int>(laplacian::windowSizeArgName,
                         R"code(Size of derivative window used in convolutions.

Window size must be odd and between 3 and 23.)code",
                         std::vector<int>{laplacian::defaultWindowSize}, true)
    .AddOptionalArg<std::vector<int>>(laplacian::smoothingSizeArgName,
                                      R"code(Size of smoothing window used in convolutions.

Smoothing size must be odd and between 1 and 23.)code",
                                      laplacian::smoothingSizeDefault, true)
    .AddOptionalArg<float>(laplacian::scaleArgName,
                           "Factors to manually scale partial derivatives.",
                           std::vector<float>{laplacian::scaleArgDefault}, true)
    .AddOptionalArg<bool>(
        laplacian::normalizeArgName,
        "If set to True, automatically scales partial derivatives kernels. Must be False "
        "if ``scale`` is specified.",
        laplacian::normalizeArgDefault)
    .AddOptionalArg("dtype", R"code(Output data type.

Supported type: `FLOAT`. If not set, the input type is used.)code",
                    DALI_NO_TYPE);


namespace laplacian {

template <typename Out, typename In, int axes, bool has_channels>
class LaplacianOpCpu : public OpImplBase<CPUBackend> {
 public:
  using Kernel = kernels::LaplacianCpu<Out, In, float, axes, has_channels>;
  static constexpr int ndim = Kernel::ndim;

  /**
   * @param spec  Pointer to a persistent OpSpec object,
   *              which is guaranteed to be alive for the entire lifetime of this object
   */
  explicit LaplacianOpCpu(const OpSpec* spec, const DimDesc& dim_desc)
      : spec_{*spec}, args{*spec}, dim_desc_{dim_desc}, lap_windows_{maxWindowSize} {}

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<CPUBackend>& ws) override {
    const auto& input = ws.template Input<CPUBackend>(0);
    int nsamples = input.num_samples();
    auto nthreads = ws.GetThreadPool().NumThreads();

    output_desc.resize(1);
    output_desc[0].type = type2id<Out>::value;
    output_desc[0].shape.resize(nsamples, input.shape().sample_dim());

    args.ObtainLaplacianArgs(spec_, ws, nsamples);
    windows_.resize(nsamples);

    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      const auto& window_sizes = args.GetWindowSizes(sample_idx);
      for (int i = 0; i < axes; i++) {
        for (int j = 0; j < axes; j++) {
          auto window_size = window_sizes[i][j];
          windows_[sample_idx][i][j] = i == j ? lap_windows_.GetDerivWindow(window_size) :
                                                lap_windows_.GetSmoothingWindow(window_size);
        }
      }
    }

    kmgr_.template Resize<Kernel>(nsamples);
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      // We take only last `ndim` siginificant dimensions to handle sequences as well
      auto elem_shape = input[sample_idx].shape().template last<ndim>();
      kmgr_.Setup<Kernel>(sample_idx, ctx_, elem_shape, args.GetWindowSizes(sample_idx));
      // The shape of data stays untouched
      output_desc[0].shape.set_tensor_shape(sample_idx, input[sample_idx].shape());
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
      auto priority = elem_volume * args.GetTotalWindowSizes(sample_idx);
      int seq_elements = volume(shape.begin(), shape.begin() + dim_desc_.usable_axes_start);
      int64_t stride = elem_volume;

      for (int elem_idx = 0; elem_idx < seq_elements; elem_idx++) {
        thread_pool.AddWork(
            [this, &input, &output, sample_idx, elem_idx, stride](int thread_id) {
              const auto& scales = args.GetScales(sample_idx);
              auto elem_shape = input[sample_idx].shape().template last<ndim>();
              auto in_view = TensorView<StorageCPU, const In, ndim>{
                  input[sample_idx].template data<In>() + stride * elem_idx, elem_shape};
              auto out_view = TensorView<StorageCPU, Out, ndim>{
                  output[sample_idx].template mutable_data<Out>() + stride * elem_idx, elem_shape};
              // Copy context so that the kernel instance can modify scratchpad
              auto ctx = ctx_;
              kmgr_.Run<Kernel>(sample_idx, ctx, out_view, in_view, windows_[sample_idx],
                                scales);
            },
            priority);
      }
    }
    thread_pool.RunAll();
  }

 private:
  const OpSpec& spec_;

  LaplacianArgs<axes> args;
  DimDesc dim_desc_;
  kernels::LaplacianWindows<float> lap_windows_;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;

  // windows_[i][j] is a window used in convolution along j-th axis in the i-th partial derivative
  std::vector<std::array<std::array<TensorView<StorageCPU, const float, 1>, axes>, axes>> windows_;
};

}  // namespace laplacian

template <>
bool Laplacian<CPUBackend>::SetupImpl(std::vector<OutputDesc>& output_desc,
                                      const workspace_t<CPUBackend>& ws) {
  const auto& input = ws.template Input<CPUBackend>(0);
  auto layout = input.GetLayout();
  auto dim_desc = ParseAndValidateDim(input.shape().sample_dim(), layout);
  auto dtype = dtype_ == DALI_NO_TYPE ? input.type() : dtype_;
  DALI_ENFORCE(dtype == input.type() || dtype == DALI_FLOAT,
               "Output data type must be same as input, FLOAT or skipped (defaults to input type)");

  if (!impl_ || impl_in_dtype_ != input.type() || impl_dim_desc_ != dim_desc) {
    impl_in_dtype_ = input.type();
    impl_dim_desc_ = dim_desc;

    TYPE_SWITCH(input.type(), type2id, In, LAPLACIAN_CPU_SUPPORTED_TYPES, (
      VALUE_SWITCH(dim_desc.usable_axes_count, Axes, LAPLACIAN_SUPPORTED_AXES, (
        BOOL_SWITCH(dim_desc.is_channel_last(), HasChannels, (
          if (dtype == input.type()) {
            using LaplacianSame = laplacian::LaplacianOpCpu<In, In, Axes, HasChannels>;
            impl_ = std::make_unique<LaplacianSame>(&spec_, dim_desc);
          } else {
            using LaplacianFloat = laplacian::LaplacianOpCpu<float, In, Axes, HasChannels>;
            impl_ = std::make_unique<LaplacianFloat>(&spec_, dim_desc);
          }
        )); // NOLINT
      ), DALI_FAIL("Axis count out of supported range."));  // NOLINT
    ), DALI_FAIL(make_string("Unsupported data type: ", input.type())));  // NOLINT
  }

  return impl_->SetupImpl(output_desc, ws);
}

template <>
void Laplacian<CPUBackend>::RunImpl(workspace_t<CPUBackend>& ws) {
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(Laplacian, Laplacian<CPUBackend>, CPU);

}  // namespace dali
