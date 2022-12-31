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
    .DocStr(R"code(Convolves the image with the provided filter.

The operator requires two positional arguments: the batch of samples and the batch of filters.

Sample can be an image, a video or volumetric (3D) data.
Samples can contain channels: channel-first and channel-last layouts are supported.
In case of video/sequences, the frame extent must preced the channels extent.
For example, a video with ``"FCHW"`` layout is supported, but ``"CFHW"`` samples are not.
Samples with the following types are supported:
int8, int16, uint8, uint16, float16, float32.
Please note that the intermediate type used for the computation is always float32.

For inputs with two spatial dimensions (images or video), the filters must be 2D arrays
(or a sequence of 2D arrays to be applied
:func:`per-frame<nvidia.dali.fn.per_frame>` to a video input).
For volumetric inputs, the filter must be a 3D array.
The filter values must have float32 type.

If the optional third positional argument is specified, it must be a batch of scalars.
If ``"border"`` is set to ``"constant"``, the input samples will be padded with
the corresponding scalars when convolved with the filter.
The scalars must be of the same type as the input samples.
For video/sequence input, an array of scalars can be specified to be applied
:func:`per-frame<nvidia.dali.fn.per_frame>`.

.. note::
  In fact, the operator computes a correlation, not a convolution,
  i.e. the order of filter elements is not flipped when computing product of
  the filter and the image.

)code")
    .NumInput(2, 3)
    .NumOutput(1)
    .AllowSequences()
    .AddOptionalArg("anchor",
                    R"code(Specifies position of the filter over the input.

If the filter size is ``(r, s)`` and the anchor is ``(a, b)``, the output
at position ``(x, y)`` is a product of the filter and the input rectangle spanned between the
corners: top-left ``(x - a, y - b)`` and bottom-right ``(x - a + r - 1, x - b + s - 1)``.

If the -1 (the default) is specifed, the middle (rounded down to integer)
of the filter extents is used, which, for odd sized filters, results in the filter
centered over the input.

The anchor must be, depending on the input dimensionality, a 2D or 3D point whose each extent lies
within filter boundaries (``[0, ..., filter_extent - 1]``). The ordering of anchor's extents
corresponds to the order of filter's extents.

The parameter is ignored in ``"valid"`` mode.
.)code",
                    std::vector<int>{-1}, true, true)
    .AddOptionalArg("border",
                    R"code(Controls how to handle out-of-bound filter positions over the sample.

Supported values are: ``"reflect_101"``, ``"reflect_1001"``, ``"wrap"``,
``"clamp"``, ``"constant"``.

- ``"reflect_101"`` (default), reflects the input but does not repeat the outermost
  values (``dcb|abcdefghi|hgf``).
- ``"reflect_1001"``: reflects the input including outermost values (``cba|abcdefghi|ihg``)
- ``"wrap"``: wraps the input (``ghi|abcdefghi|abc``).
- ``"clamp"``: the input is padded with outermost values (``aaa|abcdefghi|iii``).
- ``"constant"``: the input is padded with the user-provided scalar (zeros by default).
  within the sample.
)code",
                    "reflect_101")
    .AddOptionalArg("mode",
                    R"code(Supported values are: ``"same"`` and ``"valid"``.

- ``"same"`` (default): The input and output sizes are the same and ``border`` is used
  to handle out-of-bound filter positions.
- ``"valid"``: the output sample is cropped (by ``filter_extent - 1``) so that all
  filter positions lie fully within the input sample.
)code",
                    "same")
    .AddOptionalTypeArg("dtype", R"code(Output data type.
The output type can either be float or must be same as input type.
If not set, the input type is used.

.. note::
  The intermediate type used for actual computation is float32. If the output is of integral type,
  the values will be clamped to the output type range.
)code");

}  // namespace dali
