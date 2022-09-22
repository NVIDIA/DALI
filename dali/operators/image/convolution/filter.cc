// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/operators/image/convolution/filter.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

DALI_SCHEMA(experimental__Filter)
    .DocStr(R"code(Convolves the image with a provided filter.

The operator requires two positional arguments: the batch of images and the batch of filters.

Operator supports 2D images and video with both channels-first and channels-last layouts.

A filter must be a 2D array of filter coefficients or a sequence of 2D arrays to be applied
frame-wise to a video input. The coefficients must be floats.

The optional third argument should be a batch of scalars (or a sequence of scalars for
video input). If ``border_mode`` is set to ``"fill"``, the input samples will be padded with
the corresponding scalars when convolved with the filter, so that the convolution preserves
original shape of the image. Otherwise the argument is ignored.
The scalars must be of the same type as the input samples.

.. note::
  In fact, the operator computes a correlation, not a convolution,
  i.e. the order of filter elements is not mirrored when computing product of
  filter and a part of an image .

)code")
    .NumInput(2, 3)
    .NumOutput(1)
    .AllowSequences()
    .AddOptionalArg("anchor",
                    R"code(2D point lying within the filter specifying the placement of the
filter over an image. The ordering of extents corresponds to the ordering of filter's extents.
If -1 (the default) is specified for an extent, the middle of the extent is used.)code",
                    std::vector<int>{-1}, true, true)
    .AddOptionalArg("border_mode",
                    R"code(Controls how to compute convolution around the edges of the image, i.e.
when part of the filter lies outside of the image.

Supported values are: "reflect_101", "reflect_1001", "wrap", "replicate", "fill", "valid".

- ``"reflect_101"`` (default), reflects the input but does not repeat the outermost
    values (``dcb|abcdefghi|hgf``).
- ``"reflect_1001"``: reflects the input including outermost values (``cba|abcdefghi|ihg``)
- ``"wrap"``: wraps the input (``ghi|abcdefghi|abc``).
- ``"replicate"``: the input is padded with outermost values (``aaa|abcdefghi|iii``).
- ``"fill"``: the input is padded with the user-provided scalar (zeros by default).
- ``"valid"``: the output size is restricted so that the filter always lies fully
    within the sample.
)code",
                    "reflect_101")
    .AddOptionalTypeArg("dtype", R"code(Output data type.
Supported type: `FLOAT`. If not set, the input type is used.)code")
    .InputLayout(0, {"FHWC", "FCHW", "HWC", "CHW", "HW"});

}  // namespace dali
