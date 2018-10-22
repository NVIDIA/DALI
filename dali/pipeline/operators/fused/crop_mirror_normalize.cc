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


#include "dali/pipeline/operators/fused/crop_mirror_normalize.h"

namespace dali {

DALI_SCHEMA(CropMirrorNormalize)
  .DocStr(R"code(Perform fused cropping, normalization, format conversion
(NHWC to NCHW) if desired, and type casting.
Normalization takes input image and produces output using formula

..

   output = (input - mean) / std
)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("output_dtype",
      R"code(Output data type.)code", DALI_FLOAT)
  .AddOptionalArg("output_layout",
      R"code(Output tensor data layout)code", DALI_NCHW)
  .AddOptionalArg("pad_output",
      R"code(Whether to pad the output to number of channels being multiple of 4.)code",
      false)
  .AddOptionalArg("mirror",
      R"code(Mask for horizontal flip.

- `0` - do not perform horizontal flip for this image
- `1` - perform horizontal flip for this image.
)code", 0, true)
  .AddOptionalArg("image_type",
        R"code(The color space of input and output image.)code", DALI_RGB)
  .AddArg("mean",
      R"code(Mean pixel values for image normalization.)code",
      DALI_FLOAT_VEC)
  .AddArg("std",
      R"code(Standard deviation values for image normalization.)code",
      DALI_FLOAT_VEC)
  .AddParent("Crop");


}  // namespace dali
