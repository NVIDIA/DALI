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

#ifndef DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_
#define DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_

#include <vector>

#include "dali/core/any.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operators/expressions/arithmetic_meta.h"
#include "dali/pipeline/operators/expressions/expression_impl_factory.h"
#include "dali/pipeline/operators/expressions/expression_tree.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {
template <bool IsTensor, typename T>
using in_desc_t = std::conditional_t<IsTensor, const T*, T>;

// Using the distinction between the Tensors and Constant
// Otherwise void* could be used.
template <typename Result, typename Left, bool LeftIsTensor, typename Right, bool RightIsTensor>
struct GPUTileDesc {
  Result *result;
  in_desc_t<LeftIsTensor, Left> left;
  in_desc_t<RightIsTensor, Right> right;
  int64_t extent;
};

template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBin(Result *result, const Left *l, const Right *r, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < extent; i += blockDim.x * gridDim.x) {
    result[i] = meta::impl(l[i], r[i]);
  }
}

template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBin(Result *result, Left l, const Right *r, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < extent; i += blockDim.x * gridDim.x) {
    result[i] = meta::impl(l, r[i]);
  }
}

template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBin(Result *result, const Left *l, Right r, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < extent; i += blockDim.x * gridDim.x) {
    result[i] = meta::impl(l[i], r);
  }
}

template <ArithmeticOp op, typename Tile>
__global__ void ExecuteTiled(const Tile *tiles, int num_tiles) {
  const auto &tile = tiles[blockIdx.y];
  ExecuteBin<op>(tile.result, tile.left, tile.right, tile.extent);
}

dim3 GetGridLayout(int extent, int thread_num, int tiles) {
  return dim3(extent, tiles, 1);
}

// Assumes that it will get a workspace for given Backend,
// and the backend will store the inputs and outputs at given Backend.
template <ArithmeticOp op, typename Result,
          typename Left, bool LeftIsTensor,
          typename Right, bool RightIsTensor>
class ExprImplBinGPU : public ExprImplBase, ExprImplParam<GPUBackend> {
 public:
  ExprImplBinGPU() {}

  void Execute(ArgumentWorkspace &workspace, const OpSpec &spec, ExprImplContext &ctx,
               const std::vector<TileDesc> &tiles, TileRange range) override {
    std::vector<Tile> target_tiles;
    target_tiles.resize(range.end - range.begin);
    auto &ws = dynamic_cast<workspace_t<GPUBackend> &>(workspace);
    const auto &expr = dynamic_cast<const ExprFunc&>(*ctx.node);
    for (int i = range.begin; i < range.end; i++) {
      const auto &source_tile = tiles[i];
      target_tiles[i].left = ObtainInput<LeftIsTensor, Left>(expr, ws, spec, source_tile, 0);
      target_tiles[i].right = ObtainInput<RightIsTensor, Right>(expr, ws, spec, source_tile, 1);
      target_tiles[i].result = ObtainOutput<Result>(expr, ws, spec, source_tile);
      target_tiles[i].extent = source_tile.extent_size;
    }
    tiles_.Copy(target_tiles, ws.stream());
    Invoke(tiles_.data<Tile>(), target_tiles.size(), ws.stream());
  }

 private:
  using Tile = GPUTileDesc<Result, Left, LeftIsTensor, Right, RightIsTensor>;

  static void Invoke(const Tile *tiles, int num_tiles, cudaStream_t stream) {
    // TODO(klecki): TUNE THIS
    auto blocks = GetGridLayout(kBlocksX, kThreadNum, num_tiles);
    ExecuteTiled<op><<<blocks, kThreadNum, 0, stream>>>(tiles, num_tiles);
  }
  static constexpr int kThreadNum = 256;
  static constexpr int kBlocksX = 128;  // This should correspond to TypicalTileSize / kThreadNum?
  Tensor<GPUBackend> tiles_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_
