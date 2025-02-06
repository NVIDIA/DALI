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

#ifndef DALI_KERNELS_MATH_TRANSFORM_POINTS_CUH_
#define DALI_KERNELS_MATH_TRANSFORM_POINTS_CUH_

#include "dali/core/convert.h"
#include "dali/core/geom/mat.h"
#include "dali/core/format.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

template <typename OutCoord, typename InCoord, int out_pt_dim, int in_pt_dim>
struct TransformPointsSampleDesc {
  OutCoord *__restrict__ out;       // output point coordinates
  const InCoord *__restrict__ in;   // input point coordinates
  int64_t size;                     // number of points
  mat<out_pt_dim, in_pt_dim> M;
  vec<out_pt_dim> T;
};

template <typename Out, typename In, int out_pt_dim, int in_pt_dim>
__global__ void TransformPointsKernel(
      const TransformPointsSampleDesc<Out, In, out_pt_dim, in_pt_dim> *descs) {
  auto desc = descs[blockIdx.y];
  int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t start_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int64_t idx = start_idx; idx < desc.size; idx += grid_stride) {
    vec<in_pt_dim> v_in;
    #pragma unroll
    for (int c = 0; c < in_pt_dim; c++)
      v_in[c] = desc.in[idx * in_pt_dim + c];  // put the input in a vector

    vec<out_pt_dim> v_out = desc.M * v_in + desc.T;

    #pragma unroll
    for (int c = 0; c < out_pt_dim; c++)
      desc.out[idx * out_pt_dim + c] = ConvertSat<Out>(v_out[c]);  // unpack the vector and convert
  }
}


template <typename Out, typename In, int out_pt_dim, int in_pt_dim>
class TransformPointsGPU {
  using SampleDesc = TransformPointsSampleDesc<Out, In, out_pt_dim, in_pt_dim>;

 public:
  KernelRequirements Setup(KernelContext &ctx, const TensorListShape<> &in_shape) {
    KernelRequirements req;
    req.output_shapes = { GetOutputShape(in_shape) };

    int N = in_shape.num_samples();
    return req;
  }

  void Run(KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in,
           span<const mat<out_pt_dim, in_pt_dim>> M, span<const vec<out_pt_dim>> T) {
    int N = in.shape.num_samples();
    auto *host_descs = ctx.scratchpad->AllocatePinned<SampleDesc>(N);
    int64_t max_size = 0;
    for (int i = 0, i_m = 0, i_t = 0; i < N; i++) {
      host_descs[i].out = out.data[i];
      host_descs[i].in  = in.data[i];
      auto sample_shape = in.shape[i];
      host_descs[i].size  = volume(sample_shape.begin(), sample_shape.end() - 1);

      host_descs[i].M = M.empty() ? 1.0f : M[i_m];
      host_descs[i].T = T.empty() ? 0.0f : T[i_t];

      if (host_descs[i].size > max_size)
        max_size = host_descs[i].size;

      i_m += (M.size() > 1);  // if there's just one value in M, don't advance the index
      i_t += (T.size() > 1);  // if there's just one value in T, don't advance the index
    }
    auto *gpu_descs = ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(host_descs, N));
    const int block = 256;
    dim3 grid = GetGridSize(max_size, N, block);
    TransformPointsKernel<<<grid, block, 0, ctx.gpu.stream>>>(gpu_descs);
    CUDA_CALL(cudaGetLastError());
  }

 private:
  static TensorListShape<> GetOutputShape(const TensorListShape<> &in_shape) {
    int N = in_shape.num_samples();
    int D = in_shape.sample_dim();
    TensorListShape<> out_shape;
    out_shape.resize(N, D);
    TensorShape<> tshape;
    for (int i = 0; i < N; i++) {
      tshape = in_shape[i];
      DALI_ENFORCE(tshape[D-1] == in_pt_dim, make_string("Input contains points "
        "of incompatible dimensionality: ", tshape[D-1], " (expected ", in_pt_dim, ")"));
      tshape[D-1] = out_pt_dim;
      out_shape.set_tensor_shape(i, tshape);
    }
    return out_shape;
  }

  static dim3 GetGridSize(int64_t max_size, int num_samples, int block_size) {
    int max_blocks = div_ceil(max_size, block_size);
    int blocks_per_sample = div_ceil(max_blocks, 4);
    return dim3(std::min(blocks_per_sample, 1024), num_samples);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_MATH_TRANSFORM_POINTS_CUH_
