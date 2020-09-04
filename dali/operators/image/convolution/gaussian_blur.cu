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

using namespace gaussian_blur;  // NOLINT

// axes here is dimension of element processed by kernel - in case of sequence it's 1 less than the
// actual dim
template <typename Out, typename In, int axes, bool has_channels, bool is_sequence>
class GaussianBlurOpGpu : public OpImplBase<GPUBackend> {
 public:
  using Kernel = kernels::SeparableConvolutionGpu<Out, In, float, axes, has_channels, is_sequence>;
  static constexpr int ndim = Kernel::ndim;

  explicit GaussianBlurOpGpu(const OpSpec& spec, const DimDesc& dim_desc)
      : spec_(spec), batch_size_(spec.GetArgument<int>("batch_size")), dim_desc_(dim_desc) {}

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
      params_[i] = GetSampleParams<axes>(i, spec_, ws);
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

    // Create views (for the pointers, )
    auto in_view_dyn = view<const In>(input);
    auto out_view_dyn = view<Out>(output);

    // TODO(klecki): Just create it from the move(in_view_dyn.data), processed_shape
    auto in_view = reshape<ndim>(in_view_dyn, static_shape);
    auto out_view = reshape<ndim>(out_view_dyn, static_shape);

    kmgr_.Run<Kernel>(0, 0, ctx_, out_view, in_view, windows_tl_);

    // for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    //   const auto& shape = input[sample_idx].shape();
    //   auto elem_volume = volume(shape.begin() + dim_desc_.usable_axes_start, shape.end());

    //   int seq_elements = 1;
    //   int64_t stride = 0;
    //   if (dim_desc_.is_sequence) {
    //     seq_elements = volume(shape.begin(), shape.begin() + dim_desc_.usable_axes_start);
    //     stride = elem_volume;
    //   }
    //   for (int elem_idx = 0; elem_idx < seq_elements; elem_idx++) {
    //     thread_pool.AddWork(
    //         [this, &input, &output, sample_idx, elem_idx, stride](int thread_id) {
    //           auto gaussian_windows = windows_[sample_idx].GetWindows();
    //           auto elem_shape = input[sample_idx].shape().template last<ndim>();
    //           auto in_view = TensorView<StorageGPU, const In, ndim>{
    //               input[sample_idx].template data<In>() + stride * elem_idx, elem_shape};
    //           auto out_view = TensorView<StorageGPU, Out, ndim>{
    //               output[sample_idx].template mutable_data<Out>() + stride * elem_idx, elem_shape};
    //           // I need a context for that particular run (or rather matching the thread &
    //           // scratchpad)
    //           auto ctx = ctx_;
    //           kmgr_.Run<Kernel>(thread_id, sample_idx, ctx, out_view, in_view, gaussian_windows);
    //         }, elem_volume);
    //   }
    // }
    // thread_pool.RunAll();
  }

 private:
  OpSpec spec_;
  int batch_size_ = 0;
  DimDesc dim_desc_;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;

  std::vector<GaussianBlurParams<axes>> params_;
  std::vector<GaussianWindows<axes>> windows_;
  std::array<TensorListShape<1>, axes> window_shapes_;
  std::array<TensorListView<StorageCPU, const float, 1>, axes> windows_tl_;
};

template <>
bool GaussianBlur<GPUBackend>::SetupImpl(std::vector<OutputDesc>& output_desc,
                                         const workspace_t<GPUBackend>& ws) {
  const auto& input = ws.template InputRef<GPUBackend>(0);
  auto layout = input.GetLayout();
  auto dim_desc = ParseAndValidateDim(input.shape().sample_dim(), layout);
  dtype_ = dtype_ != DALI_NO_TYPE ? dtype_ : input.type().id();
  DALI_ENFORCE(dtype_ == input.type().id() || dtype_ == DALI_FLOAT,
               "Output data type must be same as input, FLOAT or skipped (defaults to input type)");

  // clang-format off
  TYPE_SWITCH(input.type().id(), type2id, In, GAUSSIAN_BLUR_SUPPORTED_TYPES, (
    VALUE_SWITCH(dim_desc.usable_axes_count, AXES, GAUSSIAN_BLUR_SUPPORTED_AXES, (
      VALUE_SWITCH(static_cast<int>(dim_desc.has_channels), HAS_CHANNELS, (0, 1), (
        VALUE_SWITCH(static_cast<int>(dim_desc.is_sequence), IS_SEQUENCE, (0, 1), (
          constexpr bool has_ch = HAS_CHANNELS;
          constexpr bool is_seq = IS_SEQUENCE;
          if (dtype_ == input.type().id()) {
            impl_ = std::make_unique<GaussianBlurOpGpu<In, In, AXES, has_ch, is_seq>>(spec_, dim_desc);
          } else {
            impl_ = std::make_unique<GaussianBlurOpGpu<float, In, AXES, has_ch, is_seq>>(spec_, dim_desc);
          }
        ), (DALI_FAIL("Got value different than {0, 1} when converting bool to int."))); // NOLINT
      ), (DALI_FAIL("Got value different than {0, 1} when converting bool to int."))); // NOLINT
    ), DALI_FAIL("Axis count out of supported range."));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT
  // clang-format on

  return impl_->SetupImpl(output_desc, ws);
}

template <>
void GaussianBlur<GPUBackend>::RunImpl(workspace_t<GPUBackend>& ws) {
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(GaussianBlur, GaussianBlur<GPUBackend>, GPU);

}  // namespace dali
