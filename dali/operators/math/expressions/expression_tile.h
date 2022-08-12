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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_TILE_H_
#define DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_TILE_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/pipeline/data/types.h"

namespace dali {

/**
 * @brief Describe a tile of data to be processed in expression evaluation.
 */
struct TileDesc {
  int sample_idx;       // id of sample inside within the batch
  int extent_idx;       // the index of tile within this sample_idx
  TensorShape<> extent_size;
  TensorShape<> tile_size;
};

inline std::ostream &operator<<(std::ostream &os, const TileDesc &v) {
  os << "{" << v.sample_idx << ", " << v.extent_idx << ", " << volume(v.extent_size) << ", " << volume(v.tile_size)
     << "}";
  return os;
}

struct TileRange {
  int begin;
  int end;
};

inline std::ostream &operator<<(std::ostream &os, const TileRange &v) {
  os << "{" << v.begin << ", " << v.end << "}";
  return os;
}

using OutputSamplePtr = void *;
using InputSamplePtr = const void *;

struct OperandData {
  InputSamplePtr data = nullptr;
  DALIDataType dtype = DALI_NO_TYPE;
  TensorShape<> shape{};
};

using ArgPack = SmallVector<OperandData, kMaxArity>;

/**
 * @brief Describe tile with pointers to output and input data for that tile.
 *
 * The pointers are intentionally stored as `void *` so we can use one tile type for all
 * possible `ExprImpl` implementations.
 * As we obtain pointers to Tensor/TensorList data, we cast them to `void *`
 * and the ExprImpl is later aware to what type should it be casted back.
 * This reduces the amount of types and compilation time significantly.
 */
struct ExtendedTileDesc {
  ExtendedTileDesc() = default;

  ExtendedTileDesc(const TileDesc &desc, OutputSamplePtr output, DALIDataType out_type,
                   const ArgPack &args)
      : desc(desc), output(output), out_type(out_type), args(args) {}
  TileDesc desc;
  OutputSamplePtr output;
  DALIDataType out_type;
  ArgPack args;
};

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_TILE_H_
