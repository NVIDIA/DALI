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

#ifndef DALI_PIPELINE_OPERATORS_ARITHMETIC_EXPRESSION_IMPL_GPU_CUH_
#define DALI_PIPELINE_OPERATORS_ARITHMETIC_EXPRESSION_IMPL_GPU_CUH_

#include <vector>

#include "dali/core/any.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operators/arithmetic/arithmetic_meta.h"
#include "dali/pipeline/operators/arithmetic/expression_impl_factory.h"
#include "dali/pipeline/operators/arithmetic/expression_tree.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"



namespace dali {

template <typename Result_, typename Input0Descriptor>
struct GPUUnTileDesc {
  using Result = Result_;
  using Result_storage = Result_ *;
  using Input0 = Input0Descriptor;
};

template <typename GPUUnTileDesc>
struct GPUUnTileStorage {
  typename GPUUnTileDesc::Result_storage result;
  typename GPUUnTileDesc::Input0::storage_type input0;
  int64_t extent;
};

template <typename Result_, typename Input0Descriptor, typename Input1Descriptor>
struct GPUBinTileDesc {
  using Result = Result_;
  using Result_storage = Result *;
  using Input0 = Input0Descriptor;
  using Input1 = Input1Descriptor;
};

template <typename GPUBinTileDesc>
struct GPUBinTileStorage {
  typename GPUBinTileDesc::Result_storage result;
  typename GPUBinTileDesc::Input0::storage_type input0;
  typename GPUBinTileDesc::Input1::storage_type input1;
  int64_t extent;
};

template <ArithmeticOp op, typename Result, typename Input>
__device__ void ExecuteUnOp(Result *result, const Input *in, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < extent; i += blockDim.x * gridDim.x) {
    result[i] = meta::impl(in[i]);
  }
}

template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOp(Result *result, const Left *l, const Right *r, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < extent; i += blockDim.x * gridDim.x) {
    result[i] = meta::impl(l[i], r[i]);
  }
}

template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOp(Result *result, Left l, const Right *r, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < extent; i += blockDim.x * gridDim.x) {
    result[i] = meta::impl(l, r[i]);
  }
}

template <ArithmeticOp op, typename Result, typename Left, typename Right>
__device__ void ExecuteBinOp(Result *result, const Left *l, Right r, int64_t extent) {
  using meta = arithm_meta<op, GPUBackend>;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < extent; i += blockDim.x * gridDim.x) {
    result[i] = meta::impl(l[i], r);
  }
}

template <ArithmeticOp op, typename Tile>
__device__ std::enable_if_t<GetOpArity(op) == 1>
ExecuteOp(const Tile &tile) {
  ExecuteUnOp<op>(tile.result, tile.input0, tile.extent);
}

template <ArithmeticOp op, typename Tile>
__device__ std::enable_if_t<GetOpArity(op) == 2>
ExecuteOp(const Tile &tile) {
  ExecuteBinOp<op>(tile.result, tile.input0, tile.input1, tile.extent);
}

template <ArithmeticOp op, typename Tile>
__global__ void ExecuteTiled(const Tile *tiles, int num_tiles) {
  const auto &tile = tiles[blockIdx.y];
  ExecuteOp<op>(tile);
}

dim3 GetGridLayout(int extent, int thread_num, int tiles) {
  return dim3(extent, tiles, 1);
}

// Assumes that it will get a workspace for given Backend,
// and the backend will store the inputs and outputs at given Backend.
template <ArithmeticOp op, typename GPUTileDesc, template <typename> class GPUTileStorage>
class ExpressionImplBinGPU : public ExpressionImplBase, ExpressionImplParam<GPUBackend> {
 public:
  ExpressionImplBinGPU() {}

  void Execute(ArgumentWorkspace &workspace, const OpSpec &spec, ExpressionImplContext &ctx,
               const std::vector<TileDesc> &tiles, TileRange range) override {
    std::vector<Tile> target_tiles;
    target_tiles.resize(range.end - range.begin);
    auto &ws = dynamic_cast<workspace_t<GPUBackend> &>(workspace);
    const auto &expr = *ctx.node;
    for (int i = range.begin; i < range.end; i++) {
      const auto &source_tile = tiles[i];
      FillInputs<GetOpArity(op)>(target_tiles[i], expr, ws, spec, source_tile);
      target_tiles[i].result = ObtainOutput<typename GPUTileDesc::Result>(
          expr, ws, spec, source_tile);
      target_tiles[i].extent = source_tile.extent_size;
    }
    tiles_.Copy(target_tiles, ws.stream());
    Invoke(tiles_.data<Tile>(), target_tiles.size(), ws.stream());
  }

 private:
  using Tile = GPUTileStorage<GPUTileDesc>;

  template <int Arity>
  std::enable_if_t<Arity == 2> FillInputs(Tile& target_tile, const ExprNode &expr,
                                          DeviceWorkspace &ws, const OpSpec &spec,
                                          const TileDesc &source_tile) {
    target_tile.input0 =
        ObtainInput<GPUTileDesc::Input0::is_tensor, typename GPUTileDesc::Input0::type>(
            expr, ws, spec, source_tile, 0);
    target_tile.input1 =
        ObtainInput<GPUTileDesc::Input1::is_tensor, typename GPUTileDesc::Input1::type>(
            expr, ws, spec, source_tile, 1);
  }

  template <int Arity>
  std::enable_if_t<Arity == 1> FillInputs(Tile& target_tile, const ExprNode &expr,
                                          DeviceWorkspace &ws, const OpSpec &spec,
                                          const TileDesc &source_tile) {
    target_tile.input0 =
        ObtainInput<GPUTileDesc::Input0::is_tensor, typename GPUTileDesc::Input0::type>(
            expr, ws, spec, source_tile, 0);
  }

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

#endif  // DALI_PIPELINE_OPERATORS_ARITHMETIC_EXPRESSION_IMPL_GPU_CUH_
