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

#include <vector>

#include "dali/core/tensor_layout.h"
#include "dali/operators/generic/slice/slice_attr.h"

namespace dali {

DALI_SCHEMA(SliceAttr)
    .DocStr(R"code(Slice attributes placeholder)code")
    .AddOptionalArg("axes",
        R"code(Order of dimensions used for the anchor and shape slice inputs as dimension
indices.)code",
        std::vector<int>{1, 0})
    .AddOptionalArg("axis_names",
        R"code(Order of the dimensions used for the anchor and shape slice inputs,
as described in layout.

If a value is provided, ``axis_names`` will have a higher priority than ``axes``.)code",
        TensorLayout("WH"))
    .AddOptionalArg("normalized_anchor",
        R"code(Determines whether the anchor input should be interpreted as normalized
(range [0.0, 1.0]) or as absolute coordinates.

.. note::
  This argument is only relevant when anchor data type is ``float``. For integer types,
  the coordinates are always absolute.)code",
        true)
    .AddOptionalArg("normalized_shape",
        R"code(Determines whether the shape input should be interpreted as normalized
(range [0.0, 1.0]) or as absolute coordinates.

.. note::
  This argument is only relevant when anchor data type is ``float``. For integer types,
  the coordinates are always absolute.)code",
        true);

}  // namespace dali
