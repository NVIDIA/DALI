// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/fused/resize_crop_mirror.h"

namespace dali {

DALI_REGISTER_OPERATOR(ResizeCropMirror, ResizeCropMirror<CPUBackend>, CPU);

DALI_SCHEMA(ResizeCropMirrorAttr)
  .AddOptionalArg("mirror",
      R"code(Mask for horizontal flip.

- `0` - do not perform horizontal flip for this image
- `1` - perform horizontal flip for this image.
)code", 0, true)
  .AddParent("ResizeAttr");

DALI_SCHEMA(ResizeCropMirror)
  .DocStr("Perform a fused resize, crop, mirror operation. Handles both fixed"
          " and random resizing and cropping.")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("interp_type",  // TODO(michalz): Replace with ResamplingFilterAttr when ready
      R"code(Type of interpolation used.)code",
      DALI_INTERP_LINEAR)  .AddParent("Crop")
  .AddParent("ResizeCropMirrorAttr")
  .EnforceInputLayout(DALI_NHWC);

DALI_REGISTER_OPERATOR(FastResizeCropMirror, FastResizeCropMirror<CPUBackend>, CPU);

DALI_SCHEMA(FastResizeCropMirror)
  .DocStr(
      "Perform a fused resize, crop, mirror operation. Handles both fixed "
      "and random resizing and cropping. Backprojects the desired crop "
      "through the resize operation to reduce the amount of work performed.")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddParent("ResizeCropMirror")
  .EnforceInputLayout(DALI_NHWC);

}  // namespace dali
