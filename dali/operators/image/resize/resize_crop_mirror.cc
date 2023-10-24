// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/resize/resize_crop_mirror.h"

namespace dali {

DALI_SCHEMA(ResizeCropMirrorAttr)
  .AddOptionalArg("mirror",
      R"code(Mask for flipping

Supported values:

- `0` - No flip
- `1` - Horizontal flip
- `2` - Vertical flip
- `4` - Depthwise flip
- any bitwise combination of the above)code", 0, false)
  .AddParent("ResizeAttr")
  .AddParent("CropAttr");

DALI_SCHEMA(ResizeCropMirror)
  .DocStr(R"code(Performs a fused resize, crop, mirror operation

This operator resizes a region of interest of the input image to the desired size,
optionally flipping the image.
.)code")
  .NumInput(1)
  .NumOutput(1)
  .SupportVolumetric()
  .AllowSequences()
  .AddParent("ResizeCropMirrorAttr")
  .AddParent("ResamplingFilterAttr")
  .InputLayout(0, {"HWC",  "FHWC",  "CHW",  "FCHW",  "CFHW" ,
                   "DHWC", "FDHWC", "CDHW", "FCDHW", "CFDHW"  });

DALI_SCHEMA(FastResizeCropMirror)
  .DocStr(R"code(Legacy alias for ResizedCropMirror.)code")
  .NumInput(1)
  .NumOutput(1)
  .SupportVolumetric()
  .AllowSequences()
  .AddParent("ResizeCropMirror")
  .InputLayout(0, {"HWC",  "FHWC",  "CHW",  "FCHW",  "CFHW" ,
                   "DHWC", "FDHWC", "CDHW", "FCDHW", "CFDHW"  });

template<typename Backend>
ResizeCropMirror<Backend>::ResizeCropMirror(const OpSpec &spec)
    : StatelessOperator<Backend>(spec)
    , ResizeBase<Backend>(spec)
    , resize_attr_(spec) {
  resample_params_.resize(num_threads_);
  InitializeBackend();
}


template <typename Backend>
bool ResizeCropMirror<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                const Workspace &ws) {
  output_desc.resize(1);
  auto &input = ws.Input<Backend>(0);

  const auto &in_shape = input.shape();
  auto in_type = input.type();
  auto in_layout = input.GetLayout();
  int N = in_shape.num_samples();

  PrepareParams(ws, in_shape, in_layout);

  auto out_type = resampling_attr_.GetOutputType(in_type);

  output_desc[0].type = out_type;
  this->SetupResize(output_desc[0].shape, out_type, in_shape, in_type,
                    make_cspan(this->resample_params_), NumSpatialDims(), FirstSpatialDim());
  return true;
}


template<>
void ResizeCropMirror<CPUBackend>::InitializeBackend() {
  InitializeCPU(num_threads_);
}

template<>
void ResizeCropMirror<GPUBackend>::InitializeBackend() {
  InitializeGPU(spec_.GetArgument<int>("minibatch_size"),
                spec_.GetArgument<int64_t>("temp_buffer_hint"));
}

template<typename Backend>
void ResizeCropMirror<Backend>::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<Backend>(0);
  auto &output = ws.Output<Backend>(0);

  this->RunResize(ws, output, input);
  output.SetLayout(input.GetLayout());
}


DALI_REGISTER_OPERATOR(ResizeCropMirror, ResizeCropMirror<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(ResizeCropMirror, ResizeCropMirror<GPUBackend>, GPU);

DALI_REGISTER_OPERATOR(FastResizeCropMirror, ResizeCropMirror<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(FastResizeCropMirror, ResizeCropMirror<GPUBackend>, GPU);


}  // namespace dali
