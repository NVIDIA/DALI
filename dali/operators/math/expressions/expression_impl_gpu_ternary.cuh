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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_TERNARY_CUH_
#define DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_TERNARY_CUH_

#include "dali/operators/math/expressions/expression_impl_gpu.cuh"

namespace dali {

/**
 * @brief Loop over tile of `extent` length and apply the ternary op
 */
template <ArithmeticOp op, typename Result, bool IsFirstTensor, bool IsSecondTensor,
          bool IsThirdTensor>
__device__ void ExecuteTernaryOp(Result *result,
                                 expression_detail::param_t<IsFirstTensor, Result> first,
                                 expression_detail::param_t<IsSecondTensor, Result> second,
                                 expression_detail::param_t<IsThirdTensor, Result> third,
                                 DALIDataType first_type, DALIDataType second_type,
                                 DALIDataType third_type, int64_t offset, int64_t extent) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t pos = offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t end_pos = offset + extent;
  for (; pos < end_pos; pos += stride) {
    result[pos] = meta_t::impl(expression_detail::Access<Result>(first, pos, first_type),
                               expression_detail::Access<Result>(second, pos, second_type),
                               expression_detail::Access<Result>(third, pos, third_type));
  }
}

/**
 * @brief Loop over tile of `extent` length, ternary op and different strides (used for broadcasting)
 */
template <ArithmeticOp op, typename Result, bool IsFirstTensor, bool IsSecondTensor,
          bool IsThirdTensor, int ndim>
__device__ void ExecuteTernaryOpND(Result *result,
                                   expression_detail::param_t<IsFirstTensor, Result> first,
                                   expression_detail::param_t<IsSecondTensor, Result> second,
                                   expression_detail::param_t<IsThirdTensor, Result> third,
                                   DALIDataType first_type,
                                   DALIDataType second_type,
                                   DALIDataType third_type,
                                   int64_t offset, int64_t extent,
                                   const fast_div<uint64_t> *strides_out,
                                   const int64_t *strides_first,
                                   const int64_t *strides_second,
                                   const int64_t *strides_third) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t block_start = offset + static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t block_step  = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t block_end = offset + extent;
  for (int64_t idx = block_start; idx < block_end; idx += block_step) {
    uint64_t idx_first = 0, idx_second = 0, idx_third = 0;
    uint64_t tmp_idx = idx;
    #pragma unroll
    for (int d = 0; d < ndim; d++) {
      int i_d = div_mod(tmp_idx, tmp_idx, strides_out[d]);
      idx_first += i_d * strides_first[d];
      idx_second += i_d * strides_second[d];
      idx_third += i_d * strides_third[d];
    }
    result[idx] = meta_t::impl(
      expression_detail::Access<Result>(first, idx_first, first_type),
      expression_detail::Access<Result>(second, idx_second, second_type),
      expression_detail::Access<Result>(third, idx_third, third_type));
  }
}

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result,
          bool IsFirstTensor, bool IsSecondTensor, bool IsThirdTensor>
__global__ void ExecuteTiledTernaryOp1D(const SampleDescGPU<3, 1> *samples, const TileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data);
  auto &arg0 = sample.args[0];
  auto &arg1 = sample.args[1];
  auto &arg2 = sample.args[2];
  ExecuteTernaryOp<op, Result, IsFirstTensor, IsSecondTensor, IsThirdTensor>(
      output,
      expression_detail::Pass<IsFirstTensor, Result>(arg0.data, arg0.dtype),
      expression_detail::Pass<IsSecondTensor, Result>(arg1.data, arg1.dtype),
      expression_detail::Pass<IsThirdTensor, Result>(arg2.data, arg2.dtype),
      arg0.dtype, arg1.dtype, arg2.dtype,
      tile.offset, tile.size);
}

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result,
          bool IsFirstTensor, bool IsSecondTensor, bool IsThirdTensor, int ndim>
__global__ void ExecuteTiledTernaryOpND(const SampleDescGPU<3, ndim> *samples,
                                        const TileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data);
  auto &arg0 = sample.args[0];
  auto &arg1 = sample.args[1];
  auto &arg2 = sample.args[2];
  ExecuteTernaryOpND<op, Result, IsFirstTensor, IsSecondTensor, IsThirdTensor, ndim>(
      output,
      expression_detail::Pass<IsFirstTensor, Result>(arg0.data, arg0.dtype),
      expression_detail::Pass<IsSecondTensor, Result>(arg1.data, arg1.dtype),
      expression_detail::Pass<IsThirdTensor, Result>(arg2.data, arg2.dtype),
      arg0.dtype, arg1.dtype, arg2.dtype, tile.offset, tile.size,
      &sample.output.strides[0], &arg0.strides[0],  &arg1.strides[0], &arg2.strides[0]);
}

template <ArithmeticOp op, typename Result, bool IsFirstTensor, bool IsSecondTensor,
          bool IsThirdTensor>
struct InvokerTernaryOp {
  static void Invoke(const SampleDescGPU<3, 1> *samples, const TileDesc *tiles, dim3 grid,
                     dim3 block, cudaStream_t stream) {
    ExecuteTiledTernaryOp1D<op, Result, IsFirstTensor, IsSecondTensor, IsThirdTensor>
        <<<grid, block, 0, stream>>>(samples, tiles);
  }

  template <int ndim>
  static void Invoke(const SampleDescGPU<3, ndim> *samples, const TileDesc *tiles, dim3 grid,
                     dim3 block, cudaStream_t stream) {
    // Otherwise we wouldn't land here
    assert(ndim > 1 && (IsFirstTensor + IsSecondTensor + IsThirdTensor) > 1);
    ExecuteTiledTernaryOpND<op, Result, IsFirstTensor, IsSecondTensor, IsThirdTensor, ndim>
        <<<grid, block, 0, stream>>>(samples, tiles);
  }
};


template <typename Invoker>
class ExprImplGPUInvokeTernary : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx, span<const SampleDesc> samples,
               span<const TileDesc> tiles) override {
    int ndim = samples[0].output.shape.sample_dim();
    VALUE_SWITCH(ndim, Dims, (1, 2, 3, 4, 5, 6), (
      ExecuteImpl<Dims>(ctx, samples, tiles);
    ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT
  }

  template <int ndim>
  void ExecuteImpl(ExprImplContext &ctx, span<const SampleDesc> samples,
                   span<const TileDesc> tiles) {
    kernels::DynamicScratchpad s({}, ctx.stream);
    assert(samples.size() > 0);
    for (int i = 0; i < samples.size(); i++) {
      assert(ndim == samples[i].output.shape.sample_dim());
      assert(kNumArgs == samples[i].args.size());
    }

    auto sample_descs =
        make_span(s.Allocate<mm::memory_kind::host, SampleDescGPU<kNumArgs, ndim>>(samples.size()),
                  samples.size());
    FillSampleDesc(sample_descs, samples);
    SampleDescGPU<kNumArgs, ndim> *samples_gpu = nullptr;
    TileDesc *tiles_gpu = nullptr;
    std::tie(samples_gpu, tiles_gpu) = s.ToContiguousGPU(ctx.stream, sample_descs, tiles);
    auto grid = GetGridLayout(kBlocksX, tiles.size());
    auto block = dim3(kThreadNum, 1, 1);
    Invoker::Invoke(samples_gpu, tiles_gpu, grid, block, ctx.stream);
  }

 private:
  // Use BinaryArithmeticOpGpuPerfTest for tuning
  static constexpr int kNumArgs = 3;
  static constexpr int kThreadNum = 256;
  static constexpr int kBlocksX = 64;
};

template <ArithmeticOp op, typename Result, bool IsFirstTensor, bool IsSecondTensor,
          bool IsThirdTensor>
using ExprImplGpuTernary = ExprImplGPUInvokeTernary<
    InvokerTernaryOp<op, Result, IsFirstTensor, IsSecondTensor, IsThirdTensor>>;

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_TERNARY_CUH_
