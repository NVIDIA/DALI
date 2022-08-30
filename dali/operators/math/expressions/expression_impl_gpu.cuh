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

inline bool NeedBroadcast(span<const ExtendedTileDesc> tiles) {
  for (const auto &tile : tiles) {
    SmallVector<const TensorListShape<>*, kMaxArity> shapes_ptrs;
    for (const auto &arg : tile.args)
      shapes_ptrs.push_back(&arg.shape);
    if (NeedBroadcast(make_span(shapes_ptrs)))
      return true;
  }
  return false;
}

struct SampleDescGPU {
  static constexpr int kMaxDims = 16;

  struct {
    void *ptr;
    fast_div<uint64_t> strides[kMaxDims];
    int64_t shape[kMaxDims];
  } out;

  struct {
    const void *ptr;
    int64_t shape[kMaxDims];
    int64_t strides[kMaxDims];
  } in_args[kMaxArity];

  int nargs;
  int ndim;
  bool needs_broadcasting;
};

/**
 * @brief Loop over tile of `extent` length
 */
template <ArithmeticOp op, typename Result, typename Input>
__device__ void ExecuteUnOp(Result *result, const Input *in, int64_t extent) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t start_ofs = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + extent;
  result += start_ofs;
  in += start_ofs;
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
__device__ void ExecuteBinOp(Result *result, const Left *l, const Right *r, int64_t extent) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t start_ofs = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + extent;
  result += start_ofs;
  l += start_ofs;
  r += start_ofs;
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
__device__ void ExecuteBinOp(Result *result, const Left *l, const Right *r, int64_t extent,
                             const fast_div<uint64_t> *strides_out,
                             const uint64_t *strides_l,
                             const uint64_t *strides_r) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t block_start = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t block_step  = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (uint64_t idx = block_start; idx < extent; idx += block_step) {
    uint64_t idx_l = 0, idx_r = 0;
    #pragma unroll
    for (int d = 0; d < Dims; d++) {
      int i_d = div_mod(idx, idx, strides_out[d]);
      idx_l += i_d * strides_l[d];
      idx_r += i_d * strides_r[d];
    }
    idx_l += idx;  // remaining dims have equal strides
    idx_r += idx;
    result[idx] = meta_t::impl(l[idx_l], r[idx_r]);
  }
}

/**
 * @brief Loop over tile of `extent` length, binary op with scalar on the left
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOp(Result *result, Left l, const Right *r, int64_t extent) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t start_ofs = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + extent;
  result += start_ofs;
  r += start_ofs;
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
__device__ void ExecuteBinOp(Result *result, const Left *l, Right r, int64_t extent) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t start_ofs = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + extent;
  result += start_ofs;
  l += start_ofs;
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
                                 DALIDataType third_type, int64_t extent) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t offset = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + extent;
  result += offset;
  while (result < tile_end) {
    *result = meta_t::impl(expression_detail::Access<Result>(first, offset, first_type),
                           expression_detail::Access<Result>(second, offset, second_type),
                           expression_detail::Access<Result>(third, offset, third_type));
    result += stride;
    offset += stride;
  }
}

/**
 * @brief Loop over tile of `extent` length, ternary op and different strides (used for broadcasting)
 */
template <ArithmeticOp op, int Dims, typename Result, bool IsFirstTensor, bool IsSecondTensor,
          bool IsThirdTensor>
__device__ void ExecuteTernaryOp(Result *result,
                                 expression_detail::param_t<IsFirstTensor, Result> first,
                                 expression_detail::param_t<IsSecondTensor, Result> second,
                                 expression_detail::param_t<IsThirdTensor, Result> third,
                                 DALIDataType first_type,
                                 DALIDataType second_type,
                                 DALIDataType third_type,
                                 int64_t extent,
                                 const fast_div<uint64_t> *strides_out,
                                 const uint64_t *strides_first,
                                 const uint64_t *strides_second,
                                 const uint64_t *strides_third) {
  using meta_t = arithm_meta<op, GPUBackend>;
  int64_t block_start = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t block_step  = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (uint64_t idx = block_start; idx < extent; idx += block_step) {
    uint64_t idx_first = 0, idx_second = 0, idx_third = 0;
    #pragma unroll
    for (int d = 0; d < Dims; d++) {
      int i_d = div_mod(idx, idx, strides_out[d]);
      idx_first += i_d * strides_first[d];
      idx_second += i_d * strides_second[d];
      idx_third += i_d * strides_third[d];
    }
    idx_first += idx;  // remaining dims have equal strides
    idx_second += idx;
    idx_third += idx;
    result[idx] = meta_t::impl(
      expression_detail::Access<Result>(first, idx_first, first_type),
      expression_detail::Access<Result>(second, idx_second, third_type),
      expression_detail::Access<Result>(third, idx_third, second_type));
  }
}

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result, typename Input>
__global__ void ExecuteTiledUnOp(const SampleDescGPU *samples, const TileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data) + tile.desc.offset;
  auto in = static_cast<const Input *>(sample.args[0].data) + tile.desc.offset;
  ExecuteUnOp<op>(output, in, volume(tile.desc.extent_size));
}

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right, bool IsLeftTensor,
          bool IsRightTensor>
__global__ void ExecuteTiledBinOp(const SampleDescGPU *samples, const ExtendedTileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data) + tile.desc.offset;
  auto left = static_cast<const Left *>(sample.args[0].data) + tile.desc.offset;
  auto right = static_cast<const Right *>(sample.args[1].data) + tile.desc.offset;
  if (!sample.needs_broadcasting) {
    ExecuteBinOp<op>(output,
                     expression_detail::Pass<IsLeftTensor>(left),
                     expression_detail::Pass<IsRightTensor>(right),
                     volume(tile.desc.extent_size));
  } else {
    ExecuteBinOp<op>(output,
                     expression_detail::Pass<IsLeftTensor>(left),
                     expression_detail::Pass<IsRightTensor>(right),
                     volume(tile.desc.extent_size),
                     sample.output.strides,
                     sample.args[0].strides,
                     sample.args[1].strides);
  }
}

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right, bool IsLeftTensor,
          bool IsRightTensor>
__global__ void ExecuteTiledBinOpND(const ExtendedTileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  const auto &sample = samples[tile.sample_idx];
  auto output = static_cast<Result *>(sample.output.data) + tile.desc.offset;
  auto left = static_cast<const Left *>(sample.args[0].data) + tile.desc.offset;
  auto right = static_cast<const Right *>(sample.args[1].data) + tile.desc.offset;
  using meta_t = arithm_meta<op, GPUBackend>;
  ExecuteBinOp<op>(output,
                   expression_detail::Pass<IsLeftTensor>(left),
                   expression_detail::Pass<IsRightTensor>(right),
                   volume(tile.desc.extent_size),
                   sample.output.strides,
                   sample.args[0].strides,
                   sample.args[1].strides);
}


/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result,
          bool IsFirstTensor, bool IsSecondTensor, bool IsThirdTensor>
__global__ void ExecuteTiledTernaryOp(const ExtendedTileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  auto output = static_cast<Result *>(tile.output.data);
  const void* first = tile.args[0].data;
  const void* second = tile.args[1].data;
  const void* third = tile.args[2].data;
  ExecuteTernaryOp<op, Result, IsFirstTensor, IsSecondTensor, IsThirdTensor>(
      output,
      expression_detail::Pass<IsFirstTensor, Result>(first, tile.args[0].dtype),
      expression_detail::Pass<IsSecondTensor, Result>(second, tile.args[1].dtype),
      expression_detail::Pass<IsThirdTensor, Result>(third, tile.args[2].dtype),
      tile.args[0].dtype, tile.args[0].dtype, tile.args[0].dtype,
      volume(tile.desc.extent_size));
}

template <ArithmeticOp op, typename Result, typename Input>
struct InvokerUnOp {
  static void Invoke(const ExtendedTileDesc *tiles, dim3 grid, dim3 block, cudaStream_t stream) {
    ExecuteTiledUnOp<op, Result, Input><<<grid, block, 0, stream>>>(tiles);
  }
};

template <ArithmeticOp op, typename Result, typename Left, typename Right, bool IsLeftTensor,
          bool IsRightTensor>
struct InvokerBinOp {
  static void Invoke(const ExtendedTileDesc *tiles, dim3 grid, dim3 block, cudaStream_t stream, bool need_bcast) {
    if (need_bcast) {
      ExecuteTiledBinOp<op, Result, Left, Right, IsLeftTensor, IsRightTensor>
          <<<grid, block, 0, stream>>>(tiles);
    } else {
      ExecuteTiledBinOp<op, Result, Left, Right, IsLeftTensor, IsRightTensor>
          <<<grid, block, 0, stream>>>(tiles);
    }
  }

  static void Invoke1D(const ExtendedTileDesc *tiles, dim3 grid, dim3 block, cudaStream_t stream) {
    ExecuteTiledBinOp<op, Result, Left, Right, IsLeftTensor, IsRightTensor>
        <<<grid, block, 0, stream>>>(tiles);

};

template <ArithmeticOp op, typename Result,
          bool IsFirstTensor, bool IsSecondTensor, bool IsThirdTensor>
struct InvokerTernaryOp {
  static void Invoke(const ExtendedTileDesc *tiles, dim3 grid, dim3 block, cudaStream_t stream) {
    ExecuteTiledTernaryOp<op, Result, IsFirstTensor, IsSecondTensor,
                          IsThirdTensor><<<grid, block, 0, stream>>>(tiles);
  }
};

inline dim3 GetGridLayout(int extent, int tiles) {
  return dim3(extent, tiles, 1);
}

template <typename Invoker>
class ExprImplGPUInvoke : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx, span<const ExtendedTileDesc> tiles) override {
    kernels::DynamicScratchpad s({}, ctx.stream);
    auto *tiles_pinned = s.ToPinned(tiles);
    auto *tiles_gpu = s.ToGPU(ctx.stream, make_span(tiles_pinned, tiles.size()));

    int nsamples = tiles.size();
    auto sample_descs = ctx.scratchpad->Allocate<mm::memory_kind::pinned, SampleDescGPU>(nsamples);
    for (int i = 0; i < nsamples; i++) {
      auto &sample = sample_descs[i];
      sample.out.ptr = 
    }
    auto grid = GetGridLayout(kBlocksX, tiles.size());
    auto block = dim3(kThreadNum, 1, 1);
    auto need_bcast = NeedBroadcast(tiles);
    Invoker::Invoke(tiles_gpu, grid, block, ctx.stream, need_bcast);
  }

  // TODO(janton): implement this
  void Execute(ExprImplContext &ctx, span<const SampleDesc> samples) override {
    DALI_FAIL("logic error");
  }

 private:
  // Use BinaryArithmeticOpGpuPerfTest for tuning
  static constexpr int kThreadNum = 256;
  static constexpr int kBlocksX = 64;
};

template <ArithmeticOp op, typename Result, typename Input>
using ExprImplGpuT = ExprImplGPUInvoke<InvokerUnOp<op, Result, Input>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuTT = ExprImplGPUInvoke<InvokerBinOp<op, Result, Left, Right, true, true>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuCT = ExprImplGPUInvoke<InvokerBinOp<op, Result, Left, Right, false, true>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuTC = ExprImplGPUInvoke<InvokerBinOp<op, Result, Left, Right, true, false>>;

template <ArithmeticOp op, typename Result,
          bool IsFirstTensor, bool IsSecondTensor, bool IsThirdTensor>
using ExprImplGpuTernary =
    ExprImplGPUInvoke<InvokerTernaryOp<op, Result, IsFirstTensor,
                                       IsSecondTensor, IsThirdTensor>>;

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_
