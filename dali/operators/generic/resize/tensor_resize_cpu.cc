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

#include <cassert>
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/generic/resize/tensor_resize.h"
#include "dali/pipeline/data/view_as_higher_ndim.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"


namespace dali {
namespace tensor_resize {

template <typename Out, typename In, int spatial_ndim>
class TensorResizeCPUImpl : public TensorResizeImplBase<CPUBackend> {
 public:
  static constexpr int ndim = spatial_ndim + 1;
  using Kernel = kernels::ResampleCPU<Out, In, spatial_ndim>;

  explicit TensorResizeCPUImpl(const OpSpec *spec)
      : spec_(*spec) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws,
                 const TensorListShape<> &sizes) override;
  void RunImpl(Workspace &ws) override;

 private:
  const OpSpec& spec_;
  TensorListShape<> sizes_;

  std::vector<kernels::ResamplingParamsND<spatial_ndim>> params_;
  kernels::KernelManager kmgr_;
};


template <typename Out, typename In, int spatial_ndim>
bool TensorResizeCPUImpl<Out, In, spatial_ndim>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                                           const Workspace &ws,
                                                           const TensorListShape<> &sizes) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);
  int nsamples = input.num_samples();
  int orig_ndim = input.sample_dim();

  TensorListView<StorageCPU, const In, ndim> in_view;
  if (orig_ndim == spatial_ndim) {
    in_view = view_as_higher_ndim<const In, ndim>(input, false);
  } else {
    if (orig_ndim != ndim)
      throw std::logic_error(
          make_string("Expected ", ndim, "-D data, got ", orig_ndim, "-D."));
    in_view = view<const In, ndim>(input);
  }

  if (sizes.sample_dim() != orig_ndim) {
    throw std::logic_error(make_string("Unexpected number of dimensions in sizes. Got ",
                                       sizes.sample_dim(), " but expected ", orig_ndim));
  }

  const auto &shape = in_view.shape;
  // TODO(janton) : axes
  // TODO(janton) : scales
  // TODO(janton) : aspect ratio policy
  // TODO(janton) : interpolation mode

  params_.resize(nsamples);
  for (int s = 0; s < nsamples; s++) {
    auto &p = params_[s];
    auto sz = sizes.tensor_shape_span(s);
    for (int d = 0; d < spatial_ndim; d++) {
      p[d].output_size = sz[d];
    }
  }

  kernels::KernelContext ctx;
  kmgr_.Resize<Kernel>(nsamples);
  output_desc.resize(1);
  output_desc[0].type = type2id<Out>::value;
  output_desc[0].shape.resize(nsamples, orig_ndim);
  for (int s = 0; s < nsamples; s++) {
    auto req = kmgr_.Setup<Kernel>(s, ctx, in_view[s], params_[s]);
    auto sh = output_desc[0].shape.tensor_shape_span(s);
    auto req_out_sh = req.output_shapes[0][0].shape;
    for (int d = 0; d < orig_ndim; d++)
      sh[d] = req_out_sh[d];
  }
  return true;
}


template <typename Out, typename In, int spatial_ndim>
void TensorResizeCPUImpl<Out, In, spatial_ndim>::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);

  TensorListView<StorageCPU, const In, ndim> in_view;
  TensorListView<StorageCPU, Out, ndim> out_view;
  if (input.sample_dim() == spatial_ndim) {
    in_view = view_as_higher_ndim<const In, ndim>(input, false);
    out_view = view_as_higher_ndim<Out, ndim>(output, false);
  } else {
    assert(input.sample_dim() == ndim);
    in_view = view<const In, ndim>(input);
    out_view = view<Out, ndim>(output);
  }

  output.SetLayout(input.GetLayout());
  int nsamples = input.num_samples();
  auto& thread_pool = ws.GetThreadPool();
  for (int s = 0; s < nsamples; s++) {
    thread_pool.AddWork(
      [&, s](int thread_id) {
        kernels::KernelContext ctx;
        kmgr_.Run<Kernel>(s, ctx, out_view[s], in_view[s], params_[s]);
      }, out_view.shape.tensor_size(s));
  }
  thread_pool.RunAll();
}


class TensorResizeCPU : public TensorResize<CPUBackend, TensorResizeCPUImpl> {
 public:
  explicit TensorResizeCPU(const OpSpec &spec)
      : TensorResize<CPUBackend, TensorResizeCPUImpl>(spec) {}
};


}  // namespace tensor_resize


DALI_SCHEMA(experimental__TensorResize)
    .DocStr(R"code(Resize tensors.)code")
    .AddOptionalTypeArg("dtype", R"code(Output data type.)code")
    .AddOptionalArg<std::string>("scales_rounding",
                                 R"code(Determines the rounding policy when using scales.

Possible values are:
* | ``"round"`` - Rounds to the nearest integer value, with halfway cases rounded away from zero.
* | ``"truncate"`` - Discards the fractional part of the number.)code",
                                 "truncate")
    .AddOptionalArg("sizes", R"code(Output size.)code", std::vector<int>{}, true)
    .AddOptionalArg("scales", R"code(Scale factors.)code", std::vector<float>{}, true)
    .AddOptionalArg("axes", R"code(Indices of dimensions that `sizes` and `scales` refer to.

By default, all dimensions are assumed.)code", std::vector<int>{})
    .NumInput(1)
    .NumOutput(1)
    .SupportVolumetric()
    .AllowSequences();


DALI_REGISTER_OPERATOR(experimental__TensorResize, tensor_resize::TensorResizeCPU, CPU);


}  // namespace dali
