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
namespace expr {

/**
 * @brief Describe a tile of data to be processed in expression evaluation.
 */
struct TileDesc {
  int sample_idx;       // id of sample inside within the batch
  int64_t offset;
  int64_t size;
};

inline std::ostream &operator<<(std::ostream &os, const TileDesc &v) {
  os << "{" << v.sample_idx << ", " << v.offset << ", " << v.size
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

/**
 * @brief Output tensor data
 */
struct OutputData {
  OutputSamplePtr data = nullptr;
  DALIDataType dtype = DALI_NO_TYPE;
  TensorShape<> shape{};
  TensorShape<> strides{};
};

/**
 * @brief Operand tensor data
 */
struct OperandData {
  InputSamplePtr data = nullptr;
  DALIDataType dtype = DALI_NO_TYPE;
  TensorShape<> shape{};
  TensorShape<> strides{};
};

/**
 * @brief Group of operands
 */
using ArgPack = SmallVector<OperandData, kMaxArity>;

/**
 * @brief Sample descriptor, including metadata about operands and output tensors
 */
struct SampleDesc {
  SampleDesc() = default;
  SampleDesc(OutputData output, const ArgPack &args)
      : output(output), args(args) {}
  OutputData output;
  ArgPack args;
};

}  // namespace expr
}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_TILE_H_
