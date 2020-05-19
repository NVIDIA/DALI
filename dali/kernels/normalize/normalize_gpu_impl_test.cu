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

#include <gtest/gtest.h>
#include <random>
#include "dali/test/device_test.h"
#include "dali/kernels/normalize/normalize_gpu_impl.cuh"

namespace dali {
namespace kernels {

template <bool calc_inv_stddev, typename Out, typename In>
void RefNormalize(
    const OutTensorCPU<Out> &out,
    const InTensorCPU<In> &in, const InTensorCPU<float> &base,
    const InTensorCPU<float> &scale,
    float global_scale, float shift, float epsilon,
    TensorShape<> &data_pos, TensorShape<> &base_pos, TensorShape<> &scale_pos, int dim) {

  int db = 0, ds = 0;
  int64_t extent = 0;
  if (dim < in.dim()) {
    db = base.shape[dim] > 1 ? 1 : 0;
    ds = scale.shape[dim] > 1 ? 1 : 0;
    extent = in.shape[dim];
  }
  if (dim >= in.dim() - 1) {  // handles both last dimension and degenerate case
    Out *optr = out(data_pos);
    const In *iptr = in(data_pos);
    const float *sptr = scale(scale_pos);
    const float *bptr = base(base_pos);
    for (int64_t i = 0, b = 0, s = 0; i < extent; i++, b += db, s += ds) {
      float mul;
      if (calc_inv_stddev) {
        float x = sptr[s] * sptr[s] + epsilon;
        mul = x ? rsqrt(x) * global_scale : 0;
      } else {
        mul = sptr[s] * global_scale;
      }
      optr[i] = ConvertSat<Out>(fma(iptr[i] - bptr[b], mul, shift));
    }
  } else {
    for (int64_t i = 0, b = 0, s = 0; i < extent; i++, b += db, s += ds) {
      data_pos[dim] = i;
      base_pos[dim] = b;
      scale_pos[dim] = s;
      RefNormalize<calc_inv_stddev>(out, in, base, scale, epsilon, global_scale, shift,
                                    data_pos, base_pos, scale_pos, dim + 1);
    }
  }
}



template <typename Out, typename In>
void RefNormalize(
    const OutTensorCPU<Out> &out,
    const InTensorCPU<In> &in, const InTensorCPU<float> &base,
    const InTensorCPU<float> &scale,
    float global_scale, float shift,
    bool calc_inv_stddev, float epsilon) {
  TensorShape<> data_pos, base_pos, scale_pos;
  int D = in.dim();
  data_pos.resize(D);
  base_pos.resize(D);
  scale_pos.resize(D);
  if (calc_inv_stddev) {
    RefNormalize<true>(out, in, base, scale, epsilon, global_scale, shift,
                       data_pos, base_pos, scale_pos, 0);
  } else {
    RefNormalize<false>(out, in, base, scale, epsilon, global_scale, shift,
                        data_pos, base_pos, scale_pos, 0);
  }
}


template <typename Out, typename In>
void RefNormalize(
    const OutListCPU<Out> &out, const InListCPU<In> &in,
    const InListCPU<float> &base, const InListCPU<float> &scale,
    float global_scale, float shift,
    bool calc_inv_stddev, float epsilon) {
  assert(out.shape == in.shape);
  int N = in.num_samples();
  int db = base.num_samples() > 1;
  int ds = scale.num_samples() > 1;
  for (int i = 0, b = 0, s = 0; i < N; i++, b += db, s += ds) {
    RefNormalize(out[i], in[i], base[b], scale[s], global_scale, shift, calc_inv_stddev, epsilon);
  }
}


template <typename Out, typename In>
void RefNormalize(
    const OutListCPU<Out> &out, const InListCPU<In> &in,
    const InListCPU<float> &base, float scale,
    float global_scale, float shift,
    bool calc_inv_stddev, float epsilon) {
  assert(out.shape == in.shape);
  int N = in.num_samples();
  int db = base.num_samples() > 1;
  TensorView<StorageCPU, float> scale_tv;
  scale_tv.data = &scale;
  for (int i = 0, b = 0; i < N; i++, b += db) {
    RefNormalize(out[i], in[i], base[b], scale_tv, global_scale, shift, calc_inv_stddev, epsilon);
  }
}

template <typename Out, typename In>
void RefNormalize(
    const OutListCPU<Out> &out, const InListCPU<In> &in,
    float base, const InListCPU<float> &scale,
    float global_scale, float shift,
    bool calc_inv_stddev, float epsilon) {
  assert(out.shape == in.shape);
  int N = in.num_samples();
  int ds = scale.num_samples() > 1;
  TensorView<StorageCPU, float> base_tv;
  base_tv.data = &base;
  for (int i = 0, s = 0; i < N; i++, s += ds) {
    RefNormalize(out[i], in[i], base_tv, scale[s], global_scale, shift, calc_inv_stddev, epsilon);
  }
}


TEST(NormalizeKernel, NonScalar) {

}

}  // namespace kernels
}  // namespace dali
