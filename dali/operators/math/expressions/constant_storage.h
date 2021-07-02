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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_CONSTANT_STORAGE_H_
#define DALI_OPERATORS_MATH_EXPRESSIONS_CONSTANT_STORAGE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/format.h"
#include "dali/core/static_switch.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/expression_tree.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

#define CONSTANT_STORAGE_ALLOWED_TYPES   \
  (bool,                                 \
  uint8_t, uint16_t, uint32_t, uint64_t, \
  int8_t, int16_t, int32_t, int64_t,     \
  float16, float, double)

/**
 * @brief Provide integral and floating point constants under `Backend` memory accessible by
 * a pointer.
 * Constructed by extracting the constants from spec arguments `integer_constants` and `real_constants`
 * based on list of constant_nodes
 *
 * @tparam Backend
 */
template <typename Backend>
class ConstantStorage {
 public:
  void Initialize(const OpSpec &spec, cudaStream_t stream,
                  const std::vector<ExprConstant *> &constant_nodes) {
    auto integers_vec = spec.HasArgument("integer_constants")
                            ? spec.GetRepeatedArgument<int>("integer_constants")
                            : std::vector<int>{};
    auto reals_vec = spec.HasArgument("real_constants")
                         ? spec.GetRepeatedArgument<float>("real_constants")
                         : std::vector<float>{};

    std::vector<ExprConstant *> integer_nodes, real_nodes;
    for (auto *node : constant_nodes) {
      if (IsIntegral(node->GetTypeId())) {
        integer_nodes.push_back(node);
      } else {
        real_nodes.push_back(node);
      }
    }
    integers_.set_pinned(false);
    reals_.set_pinned(false);
    Rewrite(integers_, integers_vec, integer_nodes, stream);
    Rewrite(reals_, reals_vec, real_nodes, stream);
  }

  const void *GetPointer(int constant_idx, DALIDataType type_id) const {
    if (IsIntegral(type_id)) {
      return integers_.template data<char>() + constant_idx * kPaddingSize;
    }
    return reals_.template data<char>() + constant_idx * kPaddingSize;
  }

 private:
  Tensor<Backend> integers_, reals_;

  template <typename T>
  void Rewrite(Tensor<GPUBackend> &result, const std::vector<T> &constants,
               const std::vector<ExprConstant *> &constant_nodes, cudaStream_t stream) {
    Tensor<CPUBackend> result_cpu;
    Rewrite(result_cpu, constants, constant_nodes);
    result.Copy(result_cpu, stream);
  }

  template <typename T>
  void Rewrite(Tensor<CPUBackend> &result, const std::vector<T> &constants,
               const std::vector<ExprConstant *> &constant_nodes, cudaStream_t = NULL) {
    result.Resize({static_cast<int64_t>(constants.size() * kPaddingSize)});
    char *data = result.mutable_data<char>();
    DALI_ENFORCE(
        constants.size() == constant_nodes.size(),
        make_string("Number of constants should match the number of nodes in expression tree. Got",
                    constants.size(), "constants passed and found", constant_nodes.size(),
                    "constant nodes in the expression tree"));
    for (auto *node : constant_nodes) {
      TYPE_SWITCH(node->GetTypeId(), type2id, Type, CONSTANT_STORAGE_ALLOWED_TYPES, (
          auto idx = node->GetConstIndex();
          auto *ptr = reinterpret_cast<Type *>(data + idx * kPaddingSize);
          *ptr = cast_const<Type>(constants[idx]);
        ), DALI_FAIL(make_string("Unsupported type: ", node->GetTypeId())););  // NOLINT
    }
  }

  template <typename T, typename U>
  std::enable_if_t<std::is_integral<T>::value == std::is_integral<U>::value, T> cast_const(U val) {
    return static_cast<T>(val);
  }

  template <typename T, typename U>
  std::enable_if_t<std::is_integral<T>::value != std::is_integral<U>::value, T> cast_const(U) {
    return {};
  }

  constexpr static int kPaddingSize = 8;  // max(sizeof(int64_t), sizeof(double))
};

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_CONSTANT_STORAGE_H_
