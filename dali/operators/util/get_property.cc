// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/util/get_property.h"
#include "dali/pipeline/data/types.h"

namespace dali {

DALI_SCHEMA(GetProperty)
    .DocStr(
        R"code(Returns a property of the tensor passed as an input.

The type of the output will depend on the ``key`` of the requested property.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg("key",
            R"code(Specifies, which property is requested.

The following properties are supported:

* ``"source_info"``: Returned type: byte-array.
                     String-like byte array, which contains information about the origin of the sample.
                     Fox example, :meth:`fn.get_property` called on tensor loaded via :meth:`fn.readers.file`
                     returns full path of the file, from which the tensor comes from.
* ``"layout"``: Returned type: byte-array
                :ref:`Data layout<layout_str_doc>` in the given Tensor.
)code",
            DALI_STRING);

DALI_REGISTER_OPERATOR(GetProperty, GetProperty<CPUBackend>, CPU)
DALI_REGISTER_OPERATOR(GetProperty, GetProperty<GPUBackend>, GPU)

}  // namespace dali
