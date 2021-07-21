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

#ifndef DALI_KERNELS_MATH_TRANSFORM_POINTS_H_
#define DALI_KERNELS_MATH_TRANSFORM_POINTS_H_

#include "dali/core/convert.h"
#include "dali/core/geom/mat.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

template <typename Out, typename In, int out_pt_dim, int in_pt_dim>
class TransformPointsCPU {
 public:
  KernelRequirements Setup(KernelContext &ctx, const TensorShape<> &in_shape) {
    KernelRequirements req;
    TensorShape<> out_shape = in_shape;
    out_shape[in_shape.size()-1] = out_pt_dim;
    req.output_shapes = { TensorListShape<>{{ out_shape }} };
    return req;
  }

  void Run(KernelContext &ctx, const OutTensorCPU<Out> &out, const InTensorCPU<In> &in,
           mat<out_pt_dim, in_pt_dim + 1> MT) {
    Run(ctx, out, in, sub<out_pt_dim, in_pt_dim>(MT), MT.col(in_pt_dim));
  }

  void Run(KernelContext &ctx, const OutTensorCPU<Out> &out, const InTensorCPU<In> &in,
           const mat<out_pt_dim, in_pt_dim> &M, const vec<out_pt_dim> &T) {
    int64_t n = volume(in.shape.begin(), in.shape.end() - 1);
    assert(in.num_elements() == n * in_pt_dim);
    assert(out.num_elements() == n * out_pt_dim);
    for (int64_t i = 0; i < n; i++) {
      vec<in_pt_dim> v_in;
      for (int c = 0; c < in_pt_dim; c++)
        v_in[c] = in.data[i * in_pt_dim + c];  // put the input in a vector

      vec<out_pt_dim> v_out = M * v_in + T;

      for (int c = 0; c < out_pt_dim; c++)
        out.data[i * out_pt_dim + c] = ConvertSat<Out>(v_out[c]);  // unpack the vector and convert
    }
  }
};


}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_MATH_TRANSFORM_POINTS_H_
