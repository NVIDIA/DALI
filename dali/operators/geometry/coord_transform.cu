// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/math/transform_points.cuh"
#include "dali/pipeline/data/views.h"

namespace dali {

template <>
template <typename OutputType, typename InputType, int out_dim, int in_dim>
void CoordTransform<GPUBackend>::RunTyped(Workspace &ws) {
  auto &in = ws.Input<GPUBackend>(0);
  auto &out = ws.Output<GPUBackend>(0);
  auto in_view = view<const InputType>(in);
  auto out_view = view<OutputType>(out);

  using Kernel = kernels::TransformPointsGPU<OutputType, InputType, out_dim, in_dim>;
  kmgr_.template Resize<Kernel>(1);

  int N = in_view.num_samples();
  auto M = GetMatrices<out_dim, in_dim>();
  auto T = GetTranslations<out_dim>();

  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  kmgr_.Setup<Kernel>(0, ctx, in_view.shape);
  kmgr_.Run<Kernel>(0, ctx, out_view, in_view, M, T);
}

DALI_REGISTER_OPERATOR(CoordTransform, CoordTransform<GPUBackend>, GPU);

}  // namespace dali
