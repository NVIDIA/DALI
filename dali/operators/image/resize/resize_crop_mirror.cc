// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
      R"(Mask for flipping

Supported values:

- `0` - No flip
- `1` - Horizontal flip
- `2` - Vertical flip
- `4` - Depthwise flip
- any bitwise combination of the above)", 0, true)
  .AddParent("ResizeAttr")
  .AddParent("CropAttr");

DALI_SCHEMA(ResizeCropMirror)
  .DocStr(R"(Performs a fused resize, crop, mirror operation.

The result of the operation is equivalent to applying ``resize``, followed by ``crop`` and ``flip``.
Internally, the operator calculates the relevant region of interest and performs a single
resizing operation on that region.
.)")
  .NumInput(1)
  .NumOutput(1)
  .SupportVolumetric()
  .AllowSequences()
  .AddParent("ResizeCropMirrorAttr")
  .AddParent("ResamplingFilterAttr")
  .InputLayout(0, {"HWC",  "FHWC",  "CHW",  "FCHW",  "CFHW" ,
                   "DHWC", "FDHWC", "CDHW", "FCDHW", "CFDHW"  });

DALI_SCHEMA(FastResizeCropMirror)
  .DocStr(R"(Legacy alias for ResizedCropMirror, with antialiasing disabled by default.)")
  .NumInput(1)
  .NumOutput(1)
  .SupportVolumetric()
  .AllowSequences()
  .Deprecate("1.32", "ResizeCropMirror")
  .AddParent("ResizeCropMirror")
  .AddOptionalArg("antialias", R"(If enabled, it applies an antialiasing filter when scaling down.

.. note::
  Nearest neighbor interpolation does not support antialiasing.)",
      false)
  .InputLayout(0, {"HWC",  "FHWC",  "CHW",  "FCHW",  "CFHW" ,
                   "DHWC", "FDHWC", "CDHW", "FCDHW", "CFDHW"  });

template<typename Backend>
ResizeCropMirror<Backend>::ResizeCropMirror(const OpSpec &spec)
    : StatelessOperator<Backend>(spec)
    , ResizeBase<Backend>(spec)
    , resize_attr_(spec) {
  InitializeBackend();
}

void ResizeCropMirrorAttr::PrepareResizeParams(
    const OpSpec &spec,
    const ArgumentWorkspace &ws,
    const TensorListShape<> &input_shape) {
  // First, proceed as with normal resize
  ResizeAttr::PrepareResizeParams(spec, ws, input_shape);
  mirror_.Acquire(spec, ws, batch_size_);

  // Then get the crop windows and back-project them
  for (int i = 0; i < batch_size_; i++) {
    CropAttr::ProcessArguments(spec, &ws, i);
    auto &params = params_[i];
    TensorShape<> resized_input_shape = input_shape[i];
    for (int d = 0; d < spatial_ndim_; d++)
      resized_input_shape[d + first_spatial_dim_] = params.dst_size[d];
    auto window = GetCropWindowGenerator(i)(resized_input_shape, layout_);
    int mirror = *mirror_[i].data;
    for (int d = 0; d < spatial_ndim_; d++) {
      auto d_spatial = d + first_spatial_dim_;
      double src_extent = params.src_hi[d] - params.src_lo[d];
      // Fun fact: it should work even if src_extent is negative,
      // i.e. the resize part already flips.
      double resize_ratio = src_extent / params.dst_size[d];
      double resize_offset = params.src_lo[d];

      double crop_lo = window.anchor[d_spatial];
      double crop_hi = window.anchor[d_spatial] + window.shape[d_spatial];

      params.src_lo[d] = crop_lo * resize_ratio + resize_offset;
      params.src_hi[d] = crop_hi * resize_ratio + resize_offset;

      bool mirror_this_dim = mirror & (1 << (spatial_ndim_ - 1 - d_spatial));
      if (mirror_this_dim)
        std::swap(params.src_lo[d], params.src_hi[d]);

      params.dst_size[d] = window.shape[d_spatial];
    }
  }
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
