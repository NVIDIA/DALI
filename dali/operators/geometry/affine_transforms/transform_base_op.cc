// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/geometry/affine_transforms/transform_base_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(TransformAttr)
  .DocStr(R"code(Base schema for affine transform generators.)code")
  .AddOptionalArg("reverse_order",
    R"code(Determines the order when combining affine transforms.

If set to False (default), the operator's affine transform will be applied to the input transform.
If set to True, the input transform will be applied to the operator's transform.

If there's no input, this argument is ignored.
)code",
    false);

}  // namespace dali
