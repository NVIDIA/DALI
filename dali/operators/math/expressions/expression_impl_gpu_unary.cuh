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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_UNARY_CUH_
#define DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_UNARY_CUH_

#include "dali/operators/math/expressions/expression_impl_gpu.cuh"

namespace dali {

/**
 * @brief Loop over tile of `extent` length
 */
template <ArithmeticOp op, typename Result, typename Input>
__device__ void ExecuteUnOp(Result *result, const Input *in, int64_t offset, int64_t extent) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t start = offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + offset + extent;
  result += start;
  in += start;
  while (result < tile_end) {
    *result = meta_t::impl(*in);
    result += stride;
    in += stride;
  }
}

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result, typename Input>
__global__ void ExecuteTiledUnOp(const SampleDescGPU<1, 1> *samples, const TileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto *output = static_cast<Result *>(sample.output.data) + tile.offset;
  const auto *in = static_cast<const Input *>(sample.args[0].data) + tile.offset;
  ExecuteUnOp<op>(output, in, tile.offset, tile.size);
}

template <ArithmeticOp op, typename Result, typename Input>
struct InvokerUnOp {
  static void Invoke(const SampleDescGPU<1, 1> *samples, const TileDesc *tiles, dim3 grid,
                     dim3 block, cudaStream_t stream) {
    ExecuteTiledUnOp<op, Result, Input><<<grid, block, 0, stream>>>(samples, tiles);
  }
};

template <typename Invoker>
class ExprImplGPUInvokeUnary : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx, span<const SampleDesc> samples,
               span<const TileDesc> tiles) override {
    kernels::DynamicScratchpad s({}, ctx.stream);
    TileDesc *tiles_gpu = nullptr;
    SampleDescGPU<1, 1> *samples_gpu = nullptr;

    assert(samples.size() > 0);
    int ndim = samples[0].output.shape.sample_dim();
    for (int i = 0; i < samples.size(); i++) {
      assert(ndim == samples[i].output.shape.sample_dim());
      assert(kNumArgs == samples[i].args.size());
    }
    auto grid = GetGridLayout(kBlocksX, tiles.size());
    auto block = dim3(kThreadNum, 1, 1);

    assert(ndim <= 1);  // should have been collapsed by now
    auto sample_descs =
        make_span(s.Allocate<mm::memory_kind::host, SampleDescGPU<kNumArgs, 1>>(samples.size()),
                  samples.size());
    FillSampleDesc(sample_descs, samples);
    std::tie(samples_gpu, tiles_gpu) = s.ToContiguousGPU(ctx.stream, sample_descs, tiles);
    Invoker::Invoke(samples_gpu, tiles_gpu, grid, block, ctx.stream);
  }

 private:
  // Use BinaryArithmeticOpGpuPerfTest for tuning
  static constexpr int kNumArgs = 1;
  static constexpr int kThreadNum = 256;
  static constexpr int kBlocksX = 64;
};


template <ArithmeticOp op, typename Result, typename Input>
using ExprImplGpuT = ExprImplGPUInvokeUnary<InvokerUnOp<op, Result, Input>>;

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_UNARY_CUH_
