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
#include "dali/operators/math/expressions/broadcasting.h"
#include "dali/operators/math/expressions/constant_storage.h"
#include "dali/operators/math/expressions/expression_tile.h"
#include "dali/operators/math/expressions/expression_tree.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace expr {

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

template <typename Backend>
inline OutputData GetOutput(const ExprFunc &func, Workspace &ws, int sample_idx) {
  auto &out = ws.Output<Backend>(0);
  void *out_ptr = out.raw_mutable_tensor(sample_idx);
  auto shape = out.shape()[sample_idx];
  TensorShape<> strides;
  kernels::CalcStrides(strides, shape);

  OutputData ret;
  ret.data = out_ptr;
  ret.dtype = out.type();
  ret.shape = shape;
  ret.strides = strides;
  return ret;
}

/**
 * @brief Type erased obtaining pointers to inputs
 */
template <typename Backend>
inline ArgPack GetArgPack(const ExprFunc &func, Workspace &ws,
                          const ConstantStorage<Backend> &st, const OpSpec &spec, int sample_idx) {
  ArgPack result;
  result.resize(func.GetSubexpressionCount());
  for (int i = 0; i < func.GetSubexpressionCount(); i++) {
    DALI_ENFORCE(func[i].GetNodeType() != NodeType::Function,
                 "Function nodes are not supported as subexpressions");
    if (func[i].GetNodeType() == NodeType::Constant) {
      const auto &constant = dynamic_cast<const ExprConstant &>(func[i]);
      result[i].data = st.GetPointer(constant.GetConstIndex(), constant.GetTypeId());
      result[i].dtype = constant.GetTypeId();
      result[i].shape = {};
      result[i].strides = {};
    } else if (func[i].GetNodeType() == NodeType::Tensor) {
      const auto &tensor = dynamic_cast<const ExprTensor &>(func[i]);
      auto input_idx = tensor.GetInputIndex();
      auto &in = ws.Input<Backend>(input_idx);
      result[i].data = in.raw_tensor(sample_idx);
      result[i].dtype = tensor.GetTypeId();
      result[i].shape = in.tensor_shape(sample_idx);
      kernels::CalcStrides(result[i].strides, result[i].shape);
    }
  }
  return result;
}

/**
 * @brief Extracts sample descriptor (pointer, shape, strides, dtype for inputs/outputs)
 */
template <typename Backend>
void ExtractSampleDescs(std::vector<SampleDesc> &out_samples,
                        const ExprFunc &func,
                        Workspace &ws, const ConstantStorage<Backend> &st,
                        const OpSpec &spec) {
  int nsamples =  ws.GetInputBatchSize(0);
  out_samples.clear();
  out_samples.reserve(nsamples);
  if (nsamples == 0)
    return;

  for (int s = 0; s < nsamples; s++) {
    out_samples.emplace_back(GetOutput<Backend>(func, ws, s), GetArgPack(func, ws, st, spec, s));

    SmallVector<TensorShape<>*, kMaxArity + 1> shape_ptrs;
    shape_ptrs.push_back(&(out_samples.back().output.shape));
    for (auto &arg : out_samples.back().args) {
      shape_ptrs.push_back(&arg.shape);
    }
    span<TensorShape<>*> shape_ptrs_span = make_span(shape_ptrs);
    SimplifyShapesForBroadcasting(make_span(shape_ptrs));
  }

  // Making sure all samples have same dimensionality and
  // at least 1D (the implementation requires it)
  int max_ndim = 1;
  for (int s = 0; s < nsamples; s++) {
    max_ndim = std::max(max_ndim, out_samples[s].output.shape.sample_dim());
    for (size_t a = 0; a < out_samples[s].args.size(); a++)
      max_ndim = std::max(max_ndim, out_samples[s].args[a].shape.sample_dim());
  }
  for (auto &out_sample : out_samples) {
    ExpandToNDims(out_sample.output.shape, max_ndim);
    kernels::CalcStrides(out_sample.output.strides, out_sample.output.shape);
    for (size_t a = 0; a < out_sample.args.size(); a++) {
      auto &arg = out_sample.args[a];
      ExpandToNDims(arg.shape, max_ndim);
      kernels::CalcStrides(arg.strides, arg.shape);
      arg.strides = StridesForBroadcasting(out_sample.output.shape, arg.shape, arg.strides);
    }
  }

  // Throws an error if more than 6 dims
  CheckBroadcastingSimplifiedDim(max_ndim);
}

/**
 * @brief Prepare data needed for execution.
 *        Fills vector of SampleDesc for every task that we have to execute, including
 *        the pointers to data, shapes, etc.
 */
template <typename Backend>
void PrepareSamplesPerTask(std::vector<std::vector<SampleDesc>> &samples_per_task,
                           const std::vector<ExprImplTask> &task_exec_order,
                           Workspace &ws,
                           const ConstantStorage<Backend> &constant_storage,
                           const OpSpec &spec) {
  int ntasks = task_exec_order.size();
  samples_per_task.resize(ntasks);
  for (int i = 0; i < ntasks; i++) {
    const auto &expr_task = task_exec_order[i];
    const auto &expr_func = dynamic_cast<const ExprFunc &>(*expr_task.ctx.node);
    ExtractSampleDescs<Backend>(samples_per_task[i], expr_func, ws, constant_storage, spec);
  }
}

/**
 * @brief Convert runtime expression tree `expr` to an executor for this expression by doing
 *        a static type switch over the `expr` data. CPU variant.
 */
std::unique_ptr<ExprImplBase> ExprImplFactory(const ExprNode &expr, CPUBackend);

/**
 * @brief Convert runtime expression tree `expr` to an executor for this expression by doing
 *        a static type switch over the `expr` data. GPU variant.
 */
std::unique_ptr<ExprImplBase> ExprImplFactory(const ExprNode &expr, GPUBackend);

struct ExprImplCache {
  template <typename Backend>
  ExprImplBase *GetExprImpl(const ExprNode &expr) {
    auto node_desc = expr.GetNodeDesc();
    auto it = cache_.find(node_desc);
    if (it != cache_.end()) {
      return it->second.get();
    }
    auto new_impl = ExprImplFactory(expr, Backend{});
    auto ptr = std::shared_ptr<ExprImplBase>(std::move(new_impl));
    cache_[node_desc] = ptr;
    return ptr.get();
  }

 private:
  std::map<std::string, std::shared_ptr<ExprImplBase>> cache_;
};

}  // namespace expr
}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_FACTORY_H_
