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

#include "dali/operators/geometry/coord_transform.h"
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
  .AddOptionalArg("dtype", R"(Data type of the output coordinates.

If an integral type is used, the output values are rounded to the nearest integer and clamped
to the dynamic range of this type.)",
    DALI_FLOAT,
    false)
  .AddParent("MTTransformAttr");

template <>
template <typename OutputType, typename InputType, int out_dim, int in_dim>
void CoordTransform<CPUBackend>::RunTyped(HostWorkspace &ws) {
  auto &in = ws.InputRef<CPUBackend>(0);
  auto &out = ws.OutputRef<CPUBackend>(0);
  auto in_view = view<const InputType>(in);
  auto out_view = view<OutputType>(out);
  auto &tp = ws.GetThreadPool();

  int nthreads = tp.NumThreads();

  using Kernel = kernels::TransformPointsCPU<OutputType, InputType, out_dim, in_dim>;
  kmgr_.template Resize<Kernel>(nthreads, nthreads);

  auto M = GetMatrices<out_dim, in_dim>();
  auto T = GetTranslations<out_dim>();

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
