// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/image/resize/resize.h"
#include <cassert>
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(Resize)
  .DocStr(R"code(Resize images.)code")
  .NumInput(1)
  .NumOutput(1)
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("save_attrs"));
  })
  .InputLayout(0, {"HWC", "FHWC" /*, "DHWC", "FDHWC" */ })
  .AddOptionalArg("save_attrs",
      R"code(Save reshape attributes for testing.)code", false)
  .AddParent("ResizeAttr")
  .AddParent("ResamplingFilterAttr");

template<>
Resize<CPUBackend>::Resize(const OpSpec &spec)
    : Operator<CPUBackend>(spec)
    , ResizeAttr(spec)
    , ResizeBase<CPUBackend>(spec) {
  resample_params_.resize(num_threads_);
  InitializeCPU(num_threads_);

  save_attrs_ = spec_.HasArgument("save_attrs");
}

template <>
void Resize<CPUBackend>::RunImpl(HostWorkspace &ws) {
  const int thread_idx = ws.thread_idx();
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);

  RunResize(ws, output, input);
  output.SetLayout(input.GetLayout());

  if (save_attrs_) {
    const auto &input_shape = input.shape();
    auto &attr_out = ws.OutputRef<CPUBackend>(1);
    const auto &attr_shape = attr_out.shape();
    assert(attr_shape.num_samples() == input_shape.num_samples() && attr_shape.sample_dim() == 1 &&
      is_uniform(attr_shape) && attr_shape[0][0] == spatial_ndim_);

    auto attr_view = view<int, 1>(attr_out);
    SaveAttrs(attr_view, input.shape());
  }
}

DALI_REGISTER_OPERATOR(Resize, Resize<CPUBackend>, CPU);

}  // namespace dali
