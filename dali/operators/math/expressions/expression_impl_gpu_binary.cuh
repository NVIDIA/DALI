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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_BINARY_CUH_
#define DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_BINARY_CUH_

#include "dali/operators/math/expressions/expression_impl_gpu.cuh"

namespace dali {
namespace expr {

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right>
__global__ void ExecuteTiledBinOpND(const SampleDescGPU<2> *samples, const TileDesc *tiles) {
  using meta_t = arithm_meta<op, GPUBackend>;

  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data);
  auto &out_strides = sample.output.strides;
  auto left = static_cast<const Left *>(sample.args[0].data);
  auto &left_strides = sample.args[0].strides;
  auto right = static_cast<const Right *>(sample.args[1].data);
  auto &right_strides = sample.args[1].strides;

  int ndim = sample.ndim;
  int64_t block_start = tile.offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t block_end = tile.offset + tile.size;
  int64_t block_step  = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t idx = block_start; idx < block_end; idx += block_step) {
    uint64_t idx_l = 0, idx_r = 0;
    uint64_t tmp_idx = idx;

    #pragma unroll 6
    for (int d = 0; d < ndim; d++) {
      int i_d = div_mod(tmp_idx, tmp_idx, out_strides[d]);
      idx_l += i_d * left_strides[d];
      idx_r += i_d * right_strides[d];
    }
    output[idx] = meta_t::impl(left[idx_l], right[idx_r]);
  }
}

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right,
          bool IsLeftTensor = false, bool IsRightTensor = false>
__global__ void ExecuteTiledBinOp1D(const SampleDescGPU<2> *samples, const TileDesc *tiles) {
  using meta_t = arithm_meta<op, GPUBackend>;

  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data);
  auto left = static_cast<const Left *>(sample.args[0].data);
  auto right = static_cast<const Right *>(sample.args[1].data);

  int64_t block_start = tile.offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t block_end = tile.offset + tile.size;
  int64_t block_step  = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t idx = block_start; idx < block_end; idx += block_step) {
    output[idx] = meta_t::impl(left[IsLeftTensor ? idx : 0], right[IsRightTensor ? idx : 0]);
  }
}

template <ArithmeticOp op, typename Result, typename Left, typename Right,
          bool IsLeftTensor, bool IsRightTensor>
struct InvokerBinOp {
  static void Invoke(const SampleDescGPU<2> *samples, const TileDesc *tiles, dim3 grid,
                     dim3 block, cudaStream_t stream, bool is_flat_idx) {
    if (!IsLeftTensor || !IsRightTensor || is_flat_idx) {
      ExecuteTiledBinOp1D<op, Result, Left, Right, IsLeftTensor, IsRightTensor>
        <<<grid, block, 0, stream>>>(samples, tiles);
    } else {
      ExecuteTiledBinOpND<op, Result, Left, Right>
        <<<grid, block, 0, stream>>>(samples, tiles);
    }
    CUDA_CALL(cudaGetLastError());
  }
};

template <typename Invoker>
class ExprImplGPUInvokeBinary : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx, span<const SampleDesc> samples,
               span<const TileDesc> tiles) override {
    ExecuteImpl<Invoker, 2>(ctx, samples, tiles);
  }
};

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuTT = ExprImplGPUInvokeBinary<InvokerBinOp<op, Result, Left, Right, true, true>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuCT = ExprImplGPUInvokeBinary<InvokerBinOp<op, Result, Left, Right, false, true>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuTC = ExprImplGPUInvokeBinary<InvokerBinOp<op, Result, Left, Right, true, false>>;

}  // namespace expr
}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_BINARY_CUH_
