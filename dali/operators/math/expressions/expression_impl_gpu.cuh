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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_
#define DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_

#include <vector>

#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/expression_impl_factory.h"
#include "dali/operators/math/expressions/expression_tree.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/core/fast_div.h"

namespace dali {

template <int nargs, int ndim>
struct SampleDescGPU {
  struct {
    void *data;
    DALIDataType dtype;
    fast_div<uint64_t> strides[ndim];
    int64_t shape[ndim];  // NOLINT[runtime/arrays]
  } output;

  struct {
    const void *data;
    DALIDataType dtype;
    int64_t shape[ndim];  // NOLINT[runtime/arrays]
    int64_t strides[ndim];  // NOLINT[runtime/arrays]
  } args[nargs];
};

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
template <ArithmeticOp op, typename Result, typename Input>
__global__ void ExecuteTiledUnOp(const SampleDescGPU<1, 1> *samples, const TileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto *output = static_cast<Result *>(sample.output.data) + tile.offset;
  const auto *in = static_cast<const Input *>(sample.args[0].data) + tile.offset;
  ExecuteUnOp<op>(output, in, tile.offset, tile.size);
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

template <int nargs, int ndim>
void FillSampleDesc(span<SampleDescGPU<nargs, ndim>> sample_descs, span<const SampleDesc> samples) {
  assert(sample_descs.size() == samples.size());
  for (int i = 0; i < samples.size(); i++) {
    auto &sample_desc = sample_descs[i];
    auto &sample = samples[i];
    sample_desc.output.data = sample.output.data;
    sample_desc.output.dtype = sample.output.dtype;
    for (int d = 0; d < ndim; d++) {
      sample_desc.output.shape[d] = sample.output.shape[d];
      sample_desc.output.strides[d] = sample.output.strides[d];
    }

    for (int operand_idx = 0; operand_idx < nargs; operand_idx++) {
      sample_desc.args[operand_idx].data = sample.args[operand_idx].data;
      sample_desc.args[operand_idx].dtype = sample.args[operand_idx].dtype;
      for (int d = 0; d < ndim; d++) {
        sample_desc.args[operand_idx].shape[d] = sample.args[operand_idx].shape[d];
        sample_desc.args[operand_idx].strides[d] = sample.args[operand_idx].strides[d];
      }
    }
  }
}

template <ArithmeticOp op, typename Result, typename Input>
struct InvokerUnOp {
  static void Invoke(const SampleDescGPU<1, 1> *samples, const TileDesc *tiles, dim3 grid,
                     dim3 block, cudaStream_t stream) {
    ExecuteTiledUnOp<op, Result, Input><<<grid, block, 0, stream>>>(samples, tiles);
  }
};

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

inline dim3 GetGridLayout(int extent, int tiles) {
  return dim3(extent, tiles, 1);
}

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


template <ArithmeticOp op, typename Result, typename Input>
using ExprImplGpuT = ExprImplGPUInvokeUnary<InvokerUnOp<op, Result, Input>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuTT = ExprImplGPUInvokeBinary<InvokerBinOp<op, Result, Left, Right, true, true>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuCT = ExprImplGPUInvokeBinary<InvokerBinOp<op, Result, Left, Right, false, true>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuTC = ExprImplGPUInvokeBinary<InvokerBinOp<op, Result, Left, Right, true, false>>;

template <ArithmeticOp op, typename Result, bool IsFirstTensor, bool IsSecondTensor,
          bool IsThirdTensor>
using ExprImplGpuTernary = ExprImplGPUInvokeTernary<
    InvokerTernaryOp<op, Result, IsFirstTensor, IsSecondTensor, IsThirdTensor>>;

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_
