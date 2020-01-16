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
  .DocStr(R"code(Rotate the image by given angle.)code")
  .NumInput(1)
  .NumOutput(1)
  .InputLayout(0, { "HWC", "DHWC" })
  .SupportVolumetric()
  .AddOptionalArg<float>("axis", "3D only: axis around which to rotate.\n"
  "The vector does not need to be normalized, but must have non-zero length.\n"
  "Reversing the vector is equivalent to changing the sign of `angle`.\n",
  std::vector<float>(), true)
  .AddArg("angle",
      "Angle, in degrees, by which the image is rotated.\n"
      "For 2D data, the rotation is counter-clockwise, assuming top-left corner at (0,0)\n"
      "For 3D data, the angle is a positive rotation around given axis", DALI_FLOAT, true)
  .AddOptionalArg("keep_size", "If `True`, original canvas size is kept. If `False` (default) "
"and `size` is not set, then the canvas size is adjusted to acommodate the rotated image with "
"least padding possible", false, false)
  .AddParent("WarpAttr");

DALI_REGISTER_OPERATOR(Rotate, Rotate<CPUBackend>, CPU);

}  // namespace dali
