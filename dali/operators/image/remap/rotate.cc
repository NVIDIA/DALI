// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/image/remap/rotate.h"

namespace dali {

DALI_SCHEMA(Rotate)
  .DocStr(R"code(Rotates the images by the specified angle.)code")
  .NumInput(1)
  .NumOutput(1)
  .InputLayout(0, { "HWC", "DHWC" })
  .SupportVolumetric()
  .AddOptionalArg<float>("axis", R"code(Applies **only** to three-dimension and is the axis
around which to rotate the image.

The vector does not need to be normalized, but it must have a non-zero length.
Reversing the vector is equivalent to changing the sign of ``angle``.
)code",
  std::vector<float>(), true)
  .AddArg("angle", R"code(Angle, in degrees, by which the image is rotated.

For two-dimensional data, the rotation is counter-clockwise, assuming the top-left corner is
at ``(0,0)``. For three-dimensional data, the ``angle`` is a positive rotation around the provided
axis.)code", DALI_FLOAT, true)
  .AddOptionalArg("keep_size", R"code(If True, original canvas size is kept.

If set to False (default), and the size is not set, the canvas size is adjusted to
accommodate the rotated image with the least padding possible.
)code", false, false)
  .AddParent("WarpAttr");

DALI_REGISTER_OPERATOR(Rotate, Rotate<CPUBackend>, CPU);

}  // namespace dali
