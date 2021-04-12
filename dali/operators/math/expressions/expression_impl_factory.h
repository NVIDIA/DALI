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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_FACTORY_H_
#define DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_FACTORY_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/small_vector.h"
#include "dali/core/static_switch.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/constant_storage.h"
#include "dali/operators/math/expressions/expression_tile.h"
#include "dali/operators/math/expressions/expression_tree.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

#define ALLOWED_UN_OPS                                                               \
  (ArithmeticOp::plus, ArithmeticOp::minus, ArithmeticOp::exp, ArithmeticOp::sqrt,   \
  ArithmeticOp::rsqrt, ArithmeticOp::cbrt, ArithmeticOp::log, ArithmeticOp::log2,    \
  ArithmeticOp::log10, ArithmeticOp::abs, ArithmeticOp::fabs, ArithmeticOp::floor,   \
  ArithmeticOp::ceil, ArithmeticOp::sin, ArithmeticOp::cos, ArithmeticOp::tan,       \
  ArithmeticOp::asin, ArithmeticOp::acos, ArithmeticOp::atan, ArithmeticOp::sinh,    \
  ArithmeticOp::cosh, ArithmeticOp::tanh, ArithmeticOp::asinh, ArithmeticOp::acosh,  \
  ArithmeticOp::atanh)

#define ALLOWED_BIN_OPS                                                                            \
  (ArithmeticOp::add, ArithmeticOp::sub, ArithmeticOp::mul, ArithmeticOp::div, ArithmeticOp::fdiv, \
  ArithmeticOp::mod, ArithmeticOp::min, ArithmeticOp::max, ArithmeticOp::pow, ArithmeticOp::fpow,  \
  ArithmeticOp::atan2, ArithmeticOp::eq, ArithmeticOp::neq, ArithmeticOp::lt, ArithmeticOp::leq,   \
  ArithmeticOp::gt, ArithmeticOp::geq, ArithmeticOp::bit_and, ArithmeticOp::bit_or,                \
  ArithmeticOp::bit_xor)

#define ALLOWED_TERNARY_OPS \
  (ArithmeticOp::clamp)

namespace expression_detail {

/**
 * @brief Pass through as a pointer to T or return the pointed value based on `as_ptr`
 */
template <bool as_ptr, typename T>
DALI_HOST_DEV std::enable_if_t<as_ptr, const T*> Pass(const T* ptr) {
  return ptr;
}

/**
 * @brief Pass through as a pointer to T or return the pointed value based on `as_ptr`
 */
template <bool as_ptr, typename T>
DALI_HOST_DEV std::enable_if_t<!as_ptr, T> Pass(const T* ptr) {
  return *ptr;
}

/**
 * @brief Pass through as a `const void *` or return the pointed value cast from `type_id` to T
 *        based on `as_ptr`
 */
template <bool as_ptr, typename T>
DALI_HOST_DEV std::enable_if_t<as_ptr, const void*> Pass(const void* ptr, DALIDataType) {
  return ptr;
}


/**
 * @brief Pass through as a `const void *` or return the pointed value cast from `type_id` to T
 *        based on `as_ptr`
 */
template <bool as_ptr, typename T>
DALI_HOST_DEV std::enable_if_t<!as_ptr, T> Pass(const void* ptr, DALIDataType type_id) {
  T result;
  TYPE_SWITCH(type_id, type2id, AccessType, ARITHMETIC_ALLOWED_TYPES, (
    const auto *access = reinterpret_cast<const AccessType*>(ptr);
    result = static_cast<T>(*access);
  ), result = {};);  // NOLINT(whitespace/parens)
  return result;
}

template <typename T>
DALI_HOST_DEV T Access(const T* ptr, int64_t idx) {
  return ptr[idx];
}

template <typename T>
DALI_HOST_DEV T Access(T value, int64_t) {
  return value;
}

template <typename T>
DALI_HOST_DEV T Access(const void* ptr, int64_t idx, DALIDataType type_id) {
  T result;
  TYPE_SWITCH(type_id, type2id, AccessType, ARITHMETIC_ALLOWED_TYPES, (
    const auto *access = reinterpret_cast<const AccessType*>(ptr);
    result = static_cast<T>(access[idx]);
  ), result = {};);  // NOLINT(whitespace/parens)
  return result;
}

template <typename T>
DALI_HOST_DEV T Access(T value, int64_t, DALIDataType) {
  return value;
}

template <bool as_ptr, typename T>
using param_t = std::conditional_t<as_ptr, const void*, T>;

}  // namespace expression_detail

struct ExprImplTask {
  ExprImplBase *impl;
  ExprImplContext ctx;
};

inline OutputSamplePtr GetOutputSamplePointer(HostWorkspace &ws, int output_idx, int sample_idx) {
  return ws.template OutputRef<CPUBackend>(output_idx)[sample_idx].raw_mutable_data();
}

inline OutputSamplePtr GetOutputSamplePointer(DeviceWorkspace &ws, int output_idx, int sample_idx) {
  return ws.template OutputRef<GPUBackend>(output_idx).raw_mutable_tensor(sample_idx);
}

inline InputSamplePtr GetInputSamplePointer(HostWorkspace &ws, int input_idx, int sample_idx) {
  return ws.template InputRef<CPUBackend>(input_idx)[sample_idx].raw_data();
}

inline InputSamplePtr GetInputSamplePointer(DeviceWorkspace &ws, int input_idx, int sample_idx) {
  return ws.template InputRef<GPUBackend>(input_idx).raw_tensor(sample_idx);
}

template <typename Backend>
inline OutputSamplePtr GetOutput(const ExprFunc &func, workspace_t<Backend> &ws, TileDesc tile) {
  return reinterpret_cast<char *>(GetOutputSamplePointer(ws, 0, tile.sample_idx)) +
         tile.tile_size * tile.extent_idx * TypeTable::GetTypeInfo(func.GetTypeId()).size();
}

/**
 * @brief Type erased obtaining pointers to inputs
 */
template <typename Backend>
inline ArgPack GetArgPack(const ExprFunc &func, workspace_t<Backend> &ws,
                          const ConstantStorage<Backend> &st, const OpSpec &spec, TileDesc tile) {
  ArgPack result;
  result.resize(func.GetSubexpressionCount());
  for (int i = 0; i < func.GetSubexpressionCount(); i++) {
    DALI_ENFORCE(func[i].GetNodeType() != NodeType::Function,
                 "Function nodes are not supported as subexpressions");
    if (IsScalarLike(func[i])) {
      if (func[i].GetNodeType() == NodeType::Constant) {
        const auto &constant = dynamic_cast<const ExprConstant &>(func[i]);
        result[i] = st.GetPointer(constant.GetConstIndex(), constant.GetTypeId());
      } else if (func[i].GetNodeType() == NodeType::Tensor) {
        // No tile offset, just take the pointer for this element as this is a scalar
        const auto &tensor = dynamic_cast<const ExprTensor &>(func[i]);
        auto input_idx = tensor.GetInputIndex();
        result[i] = GetInputSamplePointer(ws, input_idx, tile.sample_idx);
      }
    } else if (func[i].GetNodeType() == NodeType::Tensor) {
      const auto &tensor = dynamic_cast<const ExprTensor &>(func[i]);
      auto input_idx = tensor.GetInputIndex();
      const auto *ptr =
          reinterpret_cast<const char *>(GetInputSamplePointer(ws, input_idx, tile.sample_idx));
      auto tile_offset =
          tile.tile_size * tile.extent_idx * TypeTable::GetTypeInfo(tensor.GetTypeId()).size();
      result[i] = ptr + tile_offset;
    }
  }
  return result;
}

/**
 * @brief Transfor vector of TileDesc into vector of ExtendedTileDesc
 * based on the ExprFunc by extracting the input and output pointers to data
 * from workspace and constant storage.
 *
 * @param extended_tiles Output vector of ExtendedTiles for given task
 */
template <typename Backend>
void TransformDescs(std::vector<ExtendedTileDesc> &extended_tiles,
                    const std::vector<TileDesc> &tiles, const ExprFunc &func,
                    workspace_t<Backend> &ws, const ConstantStorage<Backend> &st,
                    const OpSpec &spec) {
  extended_tiles.reserve(tiles.size());
  SmallVector<DALIDataType, kMaxArity> in_types;
  in_types.resize(func.GetSubexpressionCount());
  for (int i = 0; i < func.GetSubexpressionCount(); i++) {
    in_types[i] = func[i].GetTypeId();
  }
  for (auto &tile : tiles) {
    extended_tiles.emplace_back(tile, GetOutput<Backend>(func, ws, tile),
                                GetArgPack(func, ws, st, spec, tile), func.GetTypeId(), in_types);
  }
}

/**
 * @brief Prepare vector of ExtendedTiles for every task that we have to execute, filling
 * the pointers to data.
 *
 * @param tiles_per_task  Output vectors of ExtendedTiles per every task to execute
 */
template <typename Backend>
void PrepareTilesForTasks(std::vector<std::vector<ExtendedTileDesc>> &tiles_per_task,
                          const std::vector<ExprImplTask> &task_exec_order,
                          const std::vector<TileDesc> &tiles, workspace_t<Backend> &ws,
                          const ConstantStorage<Backend> &constant_storage, const OpSpec &spec) {
  tiles_per_task.resize(task_exec_order.size());
  for (size_t i = 0; i < task_exec_order.size(); i++) {
    const auto &expr_task = task_exec_order[i];
    const auto &expr_func = dynamic_cast<const ExprFunc &>(*expr_task.ctx.node);
    tiles_per_task[i].resize(0);
    TransformDescs<Backend>(tiles_per_task[i], tiles, expr_func, ws, constant_storage, spec);
  }
}

/**
 * @brief Convert runtime expression tree `expr` to an executor for this expression by doing
 *        a static type switch over the `expr` data. CPU variant.
 *
 * @param ws Workspace to disambiguate over backend.
 */
std::unique_ptr<ExprImplBase> ExprImplFactory(const HostWorkspace &ws, const ExprNode &expr);

/**
 * @brief Convert runtime expression tree `expr` to an executor for this expression by doing
 *        a static type switch over the `expr` data. GPU variant.
 *
 * @param ws Workspace to disambiguate over backend.
 */
std::unique_ptr<ExprImplBase> ExprImplFactory(const DeviceWorkspace &ws, const ExprNode &expr);

struct ExprImplCache {
  template <typename Backend>
  ExprImplBase *GetExprImpl(const ExprNode &expr) {
    auto node_desc = expr.GetNodeDesc();
    auto it = cache_.find(node_desc);
    if (it != cache_.end()) {
      return it->second.get();
    }
    auto new_impl = ExprImplFactory(workspace_t<Backend>{}, expr);
    auto ptr = std::shared_ptr<ExprImplBase>(std::move(new_impl));
    cache_[node_desc] = ptr;
    return ptr.get();
  }

 private:
  std::map<std::string, std::shared_ptr<ExprImplBase>> cache_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_FACTORY_H_
