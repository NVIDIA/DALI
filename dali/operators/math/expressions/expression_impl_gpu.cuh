// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/data/types.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/expression_impl_factory.h"
#include "dali/operators/math/expressions/expression_tree.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

/**
 * @brief Loop over tile of `extent` length
 */
template <ArithmeticOp op, typename Result, typename Input>
__device__ void ExecuteUnOp(Result *result, const Input *in, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  int64_t start_ofs = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + extent;
  result += start_ofs;
  in += start_ofs;
  while (result < tile_end) {
    *result = meta::impl(*in);
    result += stride;
    in += stride;
  }
}

/**
 * @brief Loop over tile of `extent` length, binary op with two buffers as inputs
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOp(Result *result, const Left *l, const Right *r, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  int64_t start_ofs = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + extent;
  result += start_ofs;
  l += start_ofs;
  r += start_ofs;
  while (result < tile_end) {
    *result = meta::impl(*l, *r);
    result += stride;
    l += stride;
    r += stride;
  }
}

/**
 * @brief Loop over tile of `extent` length, binary op with scalar on the left
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOp(Result *result, Left l, const Right *r, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  int64_t start_ofs = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + extent;
  result += start_ofs;
  r += start_ofs;
  while (result < tile_end) {
    *result = meta::impl(l, *r);
    result += stride;
    r += stride;
  }
}

/**
 * @brief Loop over tile of `extent` length, binary op with scalar on the right
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOp(Result *result, const Left *l, Right r, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  int64_t start_ofs = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto *tile_end = result + extent;
  result += start_ofs;
  l += start_ofs;
  while (result < tile_end) {
    *result = meta::impl(*l, r);
    result += stride;
    l += stride;
  }
}

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result, typename Input>
__global__ void ExecuteTiledUnOp(const ExtendedTileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  auto output = static_cast<Result *>(tile.output);
  auto in = static_cast<const Input *>(tile.args[0]);
  ExecuteUnOp<op>(output, in, tile.desc.extent_size);
}

/**
 * @brief Pass as a pointer or dereference the variable
 */
template <bool pass_as_pointer>
struct argument {
  template <typename T>
  static DALI_HOST_DEV T pass(const T *ptr) {
    return *ptr;
  }
};

template <>
struct argument<true> {
  template <typename T>
  static DALI_HOST_DEV const T *pass(const T *ptr) {
    return ptr;
  }
};

/**
 * @brief Go over all tiles, unpacking them, casting to proper types and invoking loop over tile
 */
template <ArithmeticOp op, typename Result, typename Left, typename Right, bool IsLeftTensor,
          bool IsRightTensor>
__global__ void ExecuteTiledBinOp(const ExtendedTileDesc *tiles) {
  const auto &tile = tiles[blockIdx.y];
  auto *output = static_cast<Result *>(tile.output);
  const auto *left = static_cast<const Left *>(tile.args[0]);
  const auto *right = static_cast<const Right *>(tile.args[1]);
  ExecuteBinOp<op>(output, argument<IsLeftTensor>::pass(left), argument<IsRightTensor>::pass(right),
                   tile.desc.extent_size);
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
  static void Invoke(const ExtendedTileDesc *tiles, dim3 grid, dim3 block, cudaStream_t stream) {
    ExecuteTiledBinOp<op, Result, Left, Right, IsLeftTensor, IsRightTensor>
        <<<grid, block, 0, stream>>>(tiles);
  }
};

inline dim3 GetGridLayout(int extent, int tiles) {
  return dim3(extent, tiles, 1);
}

template <typename Invoker>
class ExprImplGPUInvoke : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx, const std::vector<ExtendedTileDesc> &tiles,
               TileRange range) override {
    tiles_.Copy(tiles, ctx.stream);
    auto grid = GetGridLayout(kBlocksX, tiles.size());
    auto block = dim3(kThreadNum, 1, 1);
    Invoker::Invoke(tiles_.data<ExtendedTileDesc>(), grid, block, ctx.stream);
  }

 private:
  // Use BinaryArithmeticOpGpuPerfTest for tuning
  static constexpr int kThreadNum = 256;
  static constexpr int kBlocksX = 64;
  Tensor<GPUBackend> tiles_;
};

template <ArithmeticOp op, typename Result, typename Input>
using ExprImplGpuT = ExprImplGPUInvoke<InvokerUnOp<op, Result, Input>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuTT = ExprImplGPUInvoke<InvokerBinOp<op, Result, Left, Right, true, true>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuCT = ExprImplGPUInvoke<InvokerBinOp<op, Result, Left, Right, false, true>>;

template <ArithmeticOp op, typename Result, typename Left, typename Right>
using ExprImplGpuTC = ExprImplGPUInvoke<InvokerBinOp<op, Result, Left, Right, true, false>>;

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_
