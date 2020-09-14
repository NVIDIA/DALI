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

#include "dali/operators/coord/coord_transform.h"

namespace dali {

DALI_SCHEMA(CoordTransform)
  .DocStr(R"(Applies a linear transformation to points or vectors.

The transformation has the form::

  out = M * in + T

Where `M` is a `m x n` matrix and `T` is a translation vector with `m` components.
Input must consist of n-element vectors or points and the output has `m` components.

This operator can be used for many operations. Here's the (incomplete) list:

 * applying affine transform to point clouds
 * projecting points onto a subspace
 * some color space conversions, for example RGB to YCbCr or grayscale
 * linear operations on colors, like hue rotation, brighness and contrast adjustment
)")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg<vector<float>>("M", R"(The matrix used for transforming the input vectors.
If left unspecified, identity matrix is used.

The matrix M does not need to be square - if it's not, the output vectors will have a
different number of dimensions.

If a scalar value is provided, M is assumed to be a square matrix with that value on the diagonal.
The size of the matrix is then assumed to match the number of components in the input vectors.
)",
    nullptr,  // no default value
    true)
  .AddOptionalArg<vector<float>>("T", R"(The translation vector.

If left unspecified, no translation is applied.

The number of components of this vector must match the number of rows in matrix M.
If a scalar value is provided, that value is broadcast to all components of T and the number of
components is chosen to match the number of rows in M.)", nullptr, true)
  .AddOptionalArg("dtype", R"(Data type of the output coordinates.

  If an integral type is used, the output values are rounded to nearest integer and clamped to
  the dynamic range of this type.)",
    DALI_FLOAT,
    false);



DALI_REGISTER_OPERATOR(CoordTransform, CoordTransform<CPUBackend>, CPU);

}  // namespace dali
