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

#include "dali/operators/image/remap/warp_affine.h"

namespace dali {

DALI_SCHEMA(WarpAffine)
  .DocStr(R"code(Applies an affine transformation to the images.)code")
  .NumInput(1, 2)
  .NumOutput(1)
  .InputLayout(0, { "HWC", "DHWC" })
  .SupportVolumetric()
  .AddOptionalArg<float>("matrix",
      R"code(Transform matrix.

When the ``inverse_map`` option is set to true (default), the matrix represents a destination to source mapping. 
With a list of values ``(M11, M12, M13, M21, M22, M23)``, this operation produces a new image
by using the following formula::

  dst(x,y) = src(M11 * x + M12 * y + M13, M21 * x + M22 * y + M23)

If the ``inverse_map`` option is set to false, the matrix represents a source to destination 
transform and it is inverted before applying to the formula above.

It is equivalent to OpenCV's ``warpAffine`` operation with the ``inverse_map`` argument being 
analog to the ``WARP_INVERSE_MAP`` flag.)code",
      vector<float>(), true)
  .AddOptionalArg<bool>("inverse_map", "Set to ``False`` if the given transform is a "
                        "destination to source mapping, ``True`` otherwise.", true, false)
  .AddParent("WarpAttr");

DALI_REGISTER_OPERATOR(WarpAffine, WarpAffine<CPUBackend>, CPU);

}  // namespace dali
