// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_TERNARY_CUH_
#define DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_TERNARY_CUH_

#include "dali/operators/math/expressions/expression_impl_gpu.cuh"

namespace dali {
namespace expr {

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result>
__global__ void ExecuteTiledTernaryOpND(const SampleDescGPU<3> *samples, const TileDesc *tiles) {
  using meta_t = arithm_meta<op, GPUBackend>;

  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data);
  auto &out_strides = sample.output.strides;
  auto &arg0 = sample.args[0];
  auto &arg1 = sample.args[1];
  auto &arg2 = sample.args[2];

  int ndim = sample.ndim;
  int64_t block_start = tile.offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t block_step  = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t block_end = tile.offset + tile.size;
  for (int64_t idx = block_start; idx < block_end; idx += block_step) {
    uint64_t idx0 = 0, idx1 = 0, idx2 = 0;
    uint64_t tmp_idx = idx;
    #pragma unroll 6
    for (int d = 0; d < ndim; d++) {
      int i_d = div_mod(tmp_idx, tmp_idx, out_strides[d]);
      idx0 += i_d * arg0.strides[d];
      idx1 += i_d * arg1.strides[d];
      idx2 += i_d * arg2.strides[d];
    }
    output[idx] = meta_t::impl(
      expression_detail::Access<Result>(arg0.data, idx0, arg0.dtype),
      expression_detail::Access<Result>(arg1.data, idx1, arg1.dtype),
      expression_detail::Access<Result>(arg2.data, idx2, arg2.dtype));
  }
}

template <ArithmeticOp op, typename Result,
          bool IsFirstTensor, bool IsSecondTensor, bool IsThirdTensor>
__global__ void ExecuteTiledTernaryOp1D(const SampleDescGPU<3> *samples, const TileDesc *tiles) {
  using meta_t = arithm_meta<op, GPUBackend>;

  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data);
  auto &arg0 = sample.args[0];
  auto &arg1 = sample.args[1];
  auto &arg2 = sample.args[2];
  int64_t block_start = tile.offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t block_step  = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t block_end = tile.offset + tile.size;
  for (int64_t idx = block_start; idx < block_end; idx += block_step) {
    output[idx] = meta_t::impl(
      expression_detail::Access<Result>(arg0.data, IsFirstTensor ? idx : 0, arg0.dtype),
      expression_detail::Access<Result>(arg1.data, IsSecondTensor ? idx : 0, arg1.dtype),
      expression_detail::Access<Result>(arg2.data, IsThirdTensor ? idx : 0, arg2.dtype));
  }
}

template <ArithmeticOp op, typename Result,
          bool IsFirstTensor, bool IsSecondTensor, bool IsThirdTensor>
struct InvokerTernaryOp {
  static void Invoke(const SampleDescGPU<3> *samples, const TileDesc *tiles, dim3 grid,
                     dim3 block, cudaStream_t stream, bool is_flat_idx) {
    if ((!IsFirstTensor + !IsSecondTensor + !IsThirdTensor) >= 2 || is_flat_idx) {
      ExecuteTiledTernaryOp1D<op, Result, IsFirstTensor, IsSecondTensor, IsThirdTensor>
          <<<grid, block, 0, stream>>>(samples, tiles);
    } else {
      ExecuteTiledTernaryOpND<op, Result>
          <<<grid, block, 0, stream>>>(samples, tiles);
    }
    CUDA_CALL(cudaGetLastError());
  }
};

template <typename Invoker>
class ExprImplGPUInvokeTernary : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx, span<const SampleDesc> samples,
               span<const TileDesc> tiles) override {
    ExecuteImpl<Invoker, 3>(ctx, samples, tiles);
  }
};

template <ArithmeticOp op, typename Result,
          bool IsFirstTensor, bool IsSecondTensor, bool IsThirdTensor>
using ExprImplGpuTernary = ExprImplGPUInvokeTernary<
    InvokerTernaryOp<op, Result, IsFirstTensor, IsSecondTensor, IsThirdTensor>>;

}  // namespace expr
}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_TERNARY_CUH_
