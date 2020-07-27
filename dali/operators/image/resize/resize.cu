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

#include <cuda_runtime_api.h>
#include <cassert>
#include <utility>
#include <vector>

#include "dali/operators/image/resize/resize.h"

namespace dali {

template<>
Resize<GPUBackend>::Resize(const OpSpec &spec)
    : Operator<GPUBackend>(spec)
    , ResizeAttr(spec)
    , ResizeBase<GPUBackend>(spec) {
  save_attrs_ = spec_.HasArgument("save_attrs");
  InitializeGPU(spec_.GetArgument<int>("minibatch_size"));
  resample_params_.resize(batch_size_);
}

template<>
void Resize<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);

  RunResize(ws, output, input);
  output.SetLayout(input.GetLayout());

  if (save_attrs_) {
    auto &attr_out = ws.Output<GPUBackend>(1);
    const auto &attr_shape = attr_out.shape();
    assert(attr_shape.num_samples() == input.shape().num_samples() &&
           attr_shape.sample_dim() == 1 &&
           is_uniform(attr_shape) &&
           attr_shape[0][0] == spatial_ndim_);

    if (!attr_staging_.raw_data())
      attr_staging_.set_pinned(true);
    attr_staging_.ResizeLike(attr_out);
    auto attr_view = view<int, 1>(attr_staging_);
    SaveAttrs(attr_view, input.shape());
    attr_out.Copy(attr_staging_, ws.stream());
  }
}

DALI_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

}  // namespace dali
