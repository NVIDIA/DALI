// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define DALI_RESIZE_BASE_CC

#include "dali/operators/image/resize/experimental/resize.h"
#include <cassert>
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(experimental__Resize)
  .DocStr(R"code(Resize images.)code")
  .NumInput(1)
  .NumOutput(1)
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("save_attrs"));
  })
  .InputLayout(0, {"HWC",  "FHWC",  "CHW",  "FCHW",  "CFHW" ,
                   "DHWC", "FDHWC", "CDHW", "FCDHW", "CFDHW"  })
  .AddOptionalArg("save_attrs",
      R"code(Save reshape attributes for testing.)code", false)
  .AddOptionalArg<DALIImageType>("image_type", "Image type", nullptr)
  .DeprecateArg("image_type", "0.25")
  .SupportVolumetric()
  .AllowSequences()
  .AddParent("ResizeAttr")
  .AddParent("ResamplingFilterAttr");

CvCudaResize::CvCudaResize(const OpSpec &spec)
    : StatelessOperator<GPUBackend>(spec)
    , ResizeBase<GPUBackend>(spec) {
  save_attrs_ = this->spec_.HasArgument("save_attrs");
  InitializeBackend();
}

void CvCudaResize::InitializeBackend() {
  InitializeGPU(spec_.GetArgument<int>("minibatch_size"),
                spec_.GetArgument<int64_t>("temp_buffer_hint"));
}

void CvCudaResize::RunImpl(Workspace &ws) {
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
           attr_shape[0][0] == NumSpatialDims());

    if (!attr_staging_.has_data())
      attr_staging_.set_pinned(true);
    attr_staging_.Resize(attr_out.shape(), DALI_INT32);
    auto attr_view = view<int, 1>(attr_staging_);
    SaveAttrs(attr_view, input.shape());
    attr_out.Copy(attr_staging_, ws.stream());
  }
}

DALI_REGISTER_OPERATOR(experimental__Resize, CvCudaResize, GPU);

}  // namespace dali
