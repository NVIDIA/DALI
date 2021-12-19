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
            R"code(Specifies, which property is requested. The following properties are supported:

* ``"source_info"``: Returned type: byte-array.
                     Information about the origin of the sample. The actual origin may differ according
                     to the source of the data. For example, when the Tensor is read via :meth:`fn.readers.file`,
                     the ``source_info`` property will contain full path of that file. When the Tensor
                     is read by :meth:`fn.readers.webdataset`, the property will contain full path of the tar
                     archive and the index of the component in that archive. When the Tensor
                     is loaded via :meth:`~nvidia.dali.fn.external_source`, the ``source_info`` will be empty.
* ``"layout"``: Returned type: byte-array
                Data layout in the given Tensor.
)code",
            DALI_STRING);

DALI_REGISTER_OPERATOR(GetProperty, GetProperty<CPUBackend>, CPU)
DALI_REGISTER_OPERATOR(GetProperty, GetProperty<GPUBackend>, GPU)

}  // namespace dali
