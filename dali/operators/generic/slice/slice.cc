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

#include "dali/operators/generic/slice/slice.h"
#include "dali/kernels/slice/slice_cpu.h"

namespace dali {

DALI_SCHEMA(Slice)
    .DocStr(
        R"code(Extracts a subtensor, or ``slice``, with a specified shape and anchor.

Inputs must be supplied as the following separate tensors in the following order:

#. ``data``
#. ``anchor``
#. ``shape``

The ``anchor`` and ``shape`` coordinates must be in the  [0.0, 1.0] interval for normalized
coordinates, or in the image shape for absolute coordinates. The ``anchor`` and ``shape`` inputs
must provide as many dimensions as are specified with the ``axis_names`` or ``axes`` arguments.
By default, the :meth:`nvidia.dali.ops.Slice` operator uses normalized coordinates and ``WH``
order for the slice arguments.)code")
    .NumInput(3)
    .NumOutput(1)
    .InputDox(0, "data", "TensorList", R"code(Batch that contains the input data.)code")
    .InputDox(1, "anchor", "1D TensorList of float",
                 R"code(Input that contains normalized or absolute coordinates, depending on
the value of ``normalized_anchor``, for the starting point of the slice (x0, x1, x2, …).)code")
    .InputDox(2, "shape", "1D TensorList of float",
                 R"code(Input that contains normalized or absolute coordinates, depending on
the value of normalized_shape, for the dimensions of the slice (s0, s1, s2, …).)code")
    .SupportVolumetric()
    .DeprecateArg("image_type", true)  // deprecated since 0.24dev
    .AddParent("SliceBase")
    .AddParent("SliceAttr")
    .AddParent("OutOfBoundsAttr");

DALI_REGISTER_OPERATOR(Slice, Slice<CPUBackend>, CPU);

}  // namespace dali
