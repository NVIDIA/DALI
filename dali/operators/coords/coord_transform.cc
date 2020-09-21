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
#include "dali/kernels/math/transform_points.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(CoordTransform)
  .DocStr(R"(Applies a linear transformation to points or vectors.


The transformation has the form::

  out = M * in + T

Where ``M`` is a ``m x n`` matrix and ``T`` is a translation vector with `m` components.
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

The matrix ``M`` does not need to be square - if it's not, the output vectors will have a
different number of components.

If a scalar value is provided, ``M`` is assumed to be a square matrix with that value on the
diagonal. The size of the matrix is then assumed to match the number of components in the
input vectors.)",
    nullptr,  // no default value
    true)
  .AddOptionalArg<vector<float>>("T", R"(The translation vector.

If left unspecified, no translation is applied.

The number of components of this vector must match the number of rows in matrix ``M``.
If a scalar value is provided, that value is broadcast to all components of ``T`` and the number
of components is chosen to match the number of rows in ``M``.)", nullptr, true)
  .AddOptionalArg<vector<float>>("MT", R"(A block matrix [M T] which combines the arguments
``M`` and ``T``.

Providing a scalar value for this argument is equivalent to providing the same scalar for
M and leaving T unspecified.

The number of columns must be one more than the number of components in the input.
This argument is mutually exclusive with ``M`` and ``T``.)",
    nullptr,
    true)
  .AddOptionalArg("dtype", R"(Data type of the output coordinates.

If an integral type is used, the output values are rounded to the nearest integer and clamped
to the dynamic range of this type.)",
    DALI_FLOAT,
    false);

template <>
template <typename OutputType, typename InputType, int out_dim, int in_dim>
void CoordTransform<CPUBackend>::RunTyped(HostWorkspace &ws) {
  auto &in = ws.InputRef<CPUBackend>(0);
  auto &out = ws.OutputRef<CPUBackend>(0);
  out.SetLayout(in.GetLayout());
  auto in_view = view<const InputType>(in);
  auto out_view = view<OutputType>(out);
  auto &tp = ws.GetThreadPool();

  int nthreads = tp.size();

  using Kernel = kernels::TransformPointsCPU<OutputType, InputType, out_dim, in_dim>;
  kmgr_.template Resize<Kernel>(nthreads, nthreads);

  auto *M = reinterpret_cast<mat<out_dim, in_dim> *>(per_sample_mtx_.data());
  auto *T = reinterpret_cast<vec<out_dim> *>(per_sample_translation_.data());

  for (int idx = 0; idx < in_view.num_samples(); idx++) {
    tp.AddWork([&, idx](int tid) {
        kernels::KernelContext ctx;
        auto in_tensor = in_view[idx];
        auto out_tensor = out_view[idx];
        kmgr_.Setup<Kernel>(tid, ctx, in_tensor.shape);
        kmgr_.Run<Kernel>(tid, tid, ctx, out_tensor, in_tensor, M[idx], T[idx]);
      }, volume(in_view.shape.tensor_shape_span(idx)));
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(CoordTransform, CoordTransform<CPUBackend>, CPU);

}  // namespace dali
