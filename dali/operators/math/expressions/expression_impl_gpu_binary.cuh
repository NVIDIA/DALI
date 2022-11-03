// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @brief Loop over tile of `extent` length, binary op with two buffers as inputs
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOp(Result *result, const Left *l, const Right *r, int64_t offset,
                               int64_t extent) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t start = offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + offset + extent;
  result += start;
  l += start;
  r += start;
  while (result < tile_end) {
    *result = meta_t::impl(*l, *r);
    result += stride;
    l += stride;
    r += stride;
  }
}

/**
 * @brief Loop over tile of `extent` length, binary op with two buffers as inputs,
          and different strides (used for broadcasting)
 */
template <ArithmeticOp op, int Dims, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOpND(Result *result, const Left *l, const Right *r, int64_t offset,
                               int64_t extent, const fast_div<uint64_t> *strides_out,
                               const int64_t *strides_l, const int64_t *strides_r) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t block_start = offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t block_end = offset + extent;
  int64_t block_step  = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t idx = block_start; idx < block_end; idx += block_step) {
    uint64_t idx_l = 0, idx_r = 0;
    uint64_t tmp_idx = idx;
    #pragma unroll
    for (int d = 0; d < Dims; d++) {
      int i_d = div_mod(tmp_idx, tmp_idx, strides_out[d]);
      idx_l += i_d * strides_l[d];
      idx_r += i_d * strides_r[d];
    }
    result[idx] = meta_t::impl(l[idx_l], r[idx_r]);
  }
}

/**
 * @brief Loop over tile of `extent` length, binary op with scalar on the left
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOp(Result *result, Left l, const Right *r, int64_t offset,
                             int64_t extent) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t start = offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + offset + extent;
  result += start;
  r += start;
  while (result < tile_end) {
    *result = meta_t::impl(l, *r);
    result += stride;
    r += stride;
  }
}

/**
 * @brief Loop over tile of `extent` length, binary op with scalar on the right
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOp(Result *result, const Left *l, Right r, int64_t offset,
                             int64_t extent) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t start = offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + offset + extent;
  result += start;
  l += start;
  while (result < tile_end) {
    *result = meta_t::impl(*l, r);
    result += stride;
    l += stride;
  }
}

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right,
          bool IsLeftTensor, bool IsRightTensor, int ndim>
__global__ void ExecuteTiledBinOpND(const SampleDescGPU<2, ndim> *samples, const TileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data);
  auto left = static_cast<const Left *>(sample.args[0].data);
  auto right = static_cast<const Right *>(sample.args[1].data);
  ExecuteBinOpND<op, ndim>(output, left, right, tile.offset, tile.size,
                           &sample.output.strides[0], &sample.args[0].strides[0],
                           &sample.args[1].strides[0]);
}

template <ArithmeticOp op, typename Result, typename Left, typename Right,
          bool IsLeftTensor, bool IsRightTensor>
__global__ void ExecuteTiledBinOp1D(const SampleDescGPU<2, 1> *samples, const TileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data);
  auto left = static_cast<const Left *>(sample.args[0].data);
  auto right = static_cast<const Right *>(sample.args[1].data);
  ExecuteBinOp<op>(output, expression_detail::Pass<IsLeftTensor>(left),
                   expression_detail::Pass<IsRightTensor>(right), tile.offset, tile.size);
}


template <ArithmeticOp op, typename Result, typename Left, typename Right, bool IsLeftTensor,
          bool IsRightTensor>
struct InvokerBinOp {
  static void Invoke(const SampleDescGPU<2, 1> *samples, const TileDesc *tiles, dim3 grid,
                     dim3 block, cudaStream_t stream) {
    ExecuteTiledBinOp1D<op, Result, Left, Right, IsLeftTensor, IsRightTensor>
        <<<grid, block, 0, stream>>>(samples, tiles);
  }

  template <int ndim>
  static void Invoke(const SampleDescGPU<2, ndim> *samples, const TileDesc *tiles, dim3 grid,
                     dim3 block, cudaStream_t stream) {
    assert(ndim > 1 && IsLeftTensor && IsRightTensor);  // Otherwise we wouldn't land here
    ExecuteTiledBinOpND<op, Result, Left, Right, IsLeftTensor, IsRightTensor, ndim>
        <<<grid, block, 0, stream>>>(samples, tiles);
  }
};

template <typename Invoker>
class ExprImplGPUInvokeBinary : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx, span<const SampleDesc> samples,
               span<const TileDesc> tiles) override {
    int ndim = samples[0].output.shape.sample_dim();
    VALUE_SWITCH(ndim, Dims, ARITHM_OPS_ALLOWED_DIMS, (
      ExecuteImpl<Dims>(ctx, samples, tiles);
    ), assert(false););  // NOLINT
  }

  template <int ndim>
  void ExecuteImpl(ExprImplContext &ctx, span<const SampleDesc> samples,
                   span<const TileDesc> tiles) {
    kernels::DynamicScratchpad s({}, ctx.stream);
    TileDesc *tiles_gpu = nullptr;
    SampleDescGPU<kNumArgs, ndim> *samples_gpu = nullptr;

    assert(samples.size() > 0);
    for (int i = 0; i < samples.size(); i++) {
      assert(ndim == samples[i].output.shape.sample_dim());
      assert(kNumArgs == samples[i].args.size());
    }

    auto samples_cpu =
        make_span(s.Allocate<mm::memory_kind::host, SampleDescGPU<kNumArgs, ndim>>(samples.size()),
                  samples.size());
    FillSampleDesc(samples_cpu, samples);

    std::tie(samples_gpu, tiles_gpu) = s.ToContiguousGPU(ctx.stream, samples_cpu, tiles);
    auto grid = GetGridLayout(kBlocksX, tiles.size());
    auto block = dim3(kThreadNum, 1, 1);
    Invoker::Invoke(samples_gpu, tiles_gpu, grid, block, ctx.stream);
  }

 private:
  // Use BinaryArithmeticOpGpuPerfTest for tuning
  static constexpr int kNumArgs = 2;
  static constexpr int kThreadNum = 256;
  static constexpr int kBlocksX = 64;
};

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuTT = ExprImplGPUInvokeBinary<InvokerBinOp<op, Result, Left, Right, true, true>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuCT = ExprImplGPUInvokeBinary<InvokerBinOp<op, Result, Left, Right, false, true>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuTC = ExprImplGPUInvokeBinary<InvokerBinOp<op, Result, Left, Right, true, false>>;

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_BINARY_CUH_
