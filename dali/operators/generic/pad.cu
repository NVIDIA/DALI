// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <vector>
#include "dali/operators/generic/pad.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/slice/slice_gpu.cuh"

namespace dali {

template <>
bool Pad<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                const Workspace &ws) {
  output_desc.resize(1);
  const auto &input = ws.Input<GPUBackend>(0);
  auto in_shape = input.shape();
  auto in_layout = input.GetLayout();
  int ndim = in_shape.sample_dim();
  int nsamples = in_shape.num_samples();

  this->ReadArguments(spec_, ws);

  TYPE_SWITCH(input.type(), type2id, T, PAD_SUPPORTED_TYPES, (
    VALUE_SWITCH(ndim, Dims, PAD_SUPPORTED_NDIMS, (
      using Kernel = kernels::SliceGPU<T, T, Dims>;
      using Args = kernels::SliceArgs<T, Dims>;

      kernels::KernelContext ctx;
      ctx.gpu.stream = ws.stream();

      auto in_view = view<const T, Dims>(input);
      auto &kernel_sample_args = FillArgs<Args>(in_shape, in_layout);

      kmgr_.Resize<Kernel>(1);
      auto req = kmgr_.Setup<Kernel>(0, ctx, in_view, kernel_sample_args);

      output_desc[0].type = type2id<T>::value;
      output_desc[0].shape.resize(nsamples, Dims);
      output_desc[0].shape = req.output_shapes[0];
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", ndim)));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type())));  // NOLINT
  return true;
}

template <>
void Pad<GPUBackend>::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());
  int ndim = input.shape().sample_dim();
  TYPE_SWITCH(input.type(), type2id, T, PAD_SUPPORTED_TYPES, (
    VALUE_SWITCH(ndim, Dims, PAD_SUPPORTED_NDIMS, (
      using Kernel = kernels::SliceGPU<T, T, Dims>;
      using Args = kernels::SliceArgs<T, Dims>;

      auto in_view = view<const T, Dims>(input);
      auto out_view = view<T, Dims>(output);
      kernels::KernelContext ctx;
      ctx.gpu.stream = ws.stream();
      auto &kernel_sample_args = std::any_cast<std::vector<Args>&>(kernel_sample_args_);
      kmgr_.Run<Kernel>(0, ctx, out_view, in_view, kernel_sample_args);
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", ndim)));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type())));  // NOLINT
}

DALI_REGISTER_OPERATOR(Pad, Pad<GPUBackend>, GPU);

}  // namespace dali
