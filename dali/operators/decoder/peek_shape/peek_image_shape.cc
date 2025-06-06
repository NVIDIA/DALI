// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/decoder/peek_shape/peek_image_shape.h"

namespace dali {

DALI_SCHEMA(PeekImageShape)
  .DocStr(R"(Obtains the shape of the encoded image.

This operator returns the shape that an image would have after decoding.

.. note::
    This operator is useful for obtaining the shapes of images without decoding them. If the images
    are decoded in full size anyway, use :meth:`nvidia.dali.pipeline.DataNode.shape()` instead on
    the decoded images.
)")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalTypeArg("dtype",
    R"code(Data type, to which the sizes are converted.)code", DALI_INT64)
  .DeprecateArgInFavorOf("type", "dtype");  // deprecated since 1.16dev

DALI_REGISTER_OPERATOR(PeekImageShape, PeekImageShape, CPU);

}  // namespace dali
