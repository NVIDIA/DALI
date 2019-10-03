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

#ifndef DALI_PIPELINE_OPERATORS_ARITHMETIC_ARITHMETIC_H_
#define DALI_PIPELINE_OPERATORS_ARITHMETIC_ARITHMETIC_H_

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/kernels/tensor_shape_print.h"
#include "dali/kernels/type_tag.h"
#include "dali/pipeline/operators/arithmetic/arithmetic_meta.h"
#include "dali/pipeline/operators/arithmetic/expression_impl_factory.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

/**
 * @brief Cover every sample fully.
 */
inline std::tuple<std::vector<TileDesc>, std::vector<TileRange>> GetCover(
    const kernels::TensorListShape<> &shape, int num_tasks) {
  std::vector<TileDesc> descs;
  std::vector<TileRange> ranges(num_tasks, {0, 0});
  int samples_per_task = shape.num_samples() / num_tasks;
  if (!samples_per_task)
    samples_per_task = shape.num_samples();
  int previous_idx = -1;
  for (int sample_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
    int task_idx = sample_idx / samples_per_task;
    descs.push_back({sample_idx, 0, task_idx, shape[sample_idx].num_elements(),
                     shape[sample_idx].num_elements()});
    if (task_idx != previous_idx) {
      ranges[task_idx].begin = descs.size() - 1;
      ranges[task_idx].end = descs.size();
    } else {
      ranges[task_idx].end = descs.size();
    }
    previous_idx = task_idx;
  }
  return std::make_tuple(descs, ranges);
}

/**
 * @brief Divide the shape into groups of linear extents, separated evenly between threads
 */
inline std::tuple<std::vector<TileDesc>, std::vector<TileRange>> GetTiledCover(
    const kernels::TensorListShape<> &shape, int extent, int num_tasks) {
  Index total_elements = shape.num_elements();
  std::vector<TileDesc> descs;
  std::vector<TileRange> ranges(num_tasks, {0, 0});
  Index elements_per_thread = total_elements / num_tasks;
  // One extent covers all
  if (total_elements < extent || elements_per_thread == 0) {
    return GetCover(shape, num_tasks);
  }
  int task_idx = 0;
  int previous_idx = -1;
  std::vector<Index> covered_by_thread(num_tasks, 0);
  for (int sample_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
    int extent_idx = 0;
    for (Index covered = 0; covered < shape[sample_idx].num_elements(); covered += extent) {
      auto actually_covered =
          std::min(static_cast<Index>(extent), shape[sample_idx].num_elements() - covered);
      descs.push_back({sample_idx, extent_idx, task_idx, actually_covered, extent});
      // We either cover a full extent, of what is left in the sample
      covered_by_thread[task_idx] += actually_covered;
      // Update range information
      if (task_idx != previous_idx) {
        ranges[task_idx].begin = descs.size() - 1;
        ranges[task_idx].end = descs.size();
      } else {
        ranges[task_idx].end = descs.size();
      }
      previous_idx = task_idx;
      // If we covered what we need, proceed to next thread
      if (covered_by_thread[task_idx] >= elements_per_thread && task_idx < num_tasks - 1) {
        task_idx++;
      }
      extent_idx++;
    }
  }
  return std::make_tuple(descs, ranges);
}

/**
 * @brief Recurse over expression tree, fill the missing types of TensorInputs
 *
 * @param expr
 * @param ws
 * @return DALIDataType
 */
template <typename Backend>
DLL_PUBLIC DALIDataType PropagateTypes(ExprNode &expr, const workspace_t<Backend> &ws) {
  if (expr.GetNodeType() == NodeType::Constant) {
    return expr.GetTypeId();
  }
  if (expr.GetNodeType() == NodeType::Tensor) {
    auto &e = dynamic_cast<ExprTensor&>(expr);
    expr.SetTypeId(ws.template InputRef<Backend>(e.GetMappedInput()).type().id());
    return expr.GetTypeId();
  }
  if (expr.GetSubexpressionCount() == 2) {
    auto left_type = PropagateTypes<Backend>(expr[0], ws);
    auto right_type = PropagateTypes<Backend>(expr[1], ws);
    expr.SetTypeId(TypePromotion(left_type, right_type));
    return expr.GetTypeId();
  }
  DALI_FAIL("Only binary expressions are supported");
}


struct ExpressionImplTask {
  ExpressionImplBase * impl;
  ExpressionImplContext ctx;
};

template <typename Backend>
inline void CreateExecutionOrder(std::vector<ExpressionImplTask> &order, const ExprNode &expr,
                                 ExprImplCache &cache) {
  if (expr.GetNodeType() != NodeType::Function) {
    return;
  }
  for (int i = 0; i < expr.GetSubexpressionCount(); i++) {
    CreateExecutionOrder<Backend>(order, expr[i], cache);
  }
  order.push_back({cache.GetExprImpl<Backend>(expr), {&expr}});
}

template <typename Backend>
inline std::vector<ExpressionImplTask> CreateExecutionOrder(const ExprNode &expr,
                                                              ExprImplCache &cache) {
  std::vector<ExpressionImplTask> result;
  CreateExecutionOrder<Backend>(result, expr, cache);
  return result;
}

inline kernels::TensorListShape<> ShapePromotion(std::string op,
                                                 const kernels::TensorListShape<> &left,
                                                 const kernels::TensorListShape<> &right) {
  bool is_left_scalar = IsScalarLike(left);
  bool is_right_scalar = IsScalarLike(right);
  if (is_left_scalar && is_right_scalar) {
    return kernels::TensorListShape<>{{1}};
  }
  if (is_left_scalar) {
    return right;
  }
  if (is_right_scalar) {
    return left;
  }
  using std::to_string;
  DALI_ENFORCE(left == right, "Input shapes of elemenetwise arithemtic operator \"" + op +
                                  "\" do not match. Expected equal shapes, got: " + op + "(" +
                                  to_string(left) + ", " + to_string(right) + ").");
  return left;
}

template <typename Backend>
DLL_PUBLIC kernels::TensorListShape<> PropagateShapes(ExprNode &expr,
                                                      const workspace_t<Backend> &ws) {
  if (expr.GetNodeType() == NodeType::Constant) {
    expr.SetShape(kernels::TensorListShape<>{{1}});
    return expr.GetShape();
  }
  if (expr.GetNodeType() == NodeType::Tensor) {
    auto &e = dynamic_cast<ExprTensor&>(expr);
    expr.SetShape(ws.template InputRef<Backend>(e.GetMappedInput()).shape());
    return expr.GetShape();
  }
  if (expr.GetSubexpressionCount() == 2) {
    auto left_shape = PropagateShapes<Backend>(expr[0], ws);
    auto right_shape = PropagateShapes<Backend>(expr[1], ws);
    expr.SetShape(ShapePromotion(expr.GetOp(), left_shape, right_shape));
    return expr.GetShape();
  }
  DALI_FAIL("Only binary expressions are supported");
}

template <typename Backend>
class ArithmeticGenericOp : public Operator<Backend> {
 public:
  inline explicit ArithmeticGenericOp(const OpSpec &spec) : Operator<Backend>(spec) {
    expr_ = ParseExpressionString(spec.GetArgument<std::string>("expression_desc"));
    num_tasks_ = 2 * num_threads_;
  }

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    output_desc.resize(1);

    result_type_id_ = PropagateTypes<Backend>(*expr_, ws);  // can be done once

    result_shape_ = PropagateShapes<Backend>(*expr_, ws);
    AllocateIntermediateNodes();
    exec_order_ = CreateExecutionOrder<Backend>(*expr_, cache_);

    output_desc[0] = {result_shape_, TypeTable::GetTypeInfo(result_type_id_)};
    // TODO(klecki): cover for GPU should be whole batch in one go/non linear tensor list?
    std::tie(tile_cover_, tile_range_) = GetTiledCover(result_shape_, kExtent, num_tasks_);
    return true;
  }

  using Operator<Backend>::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override;

 private:
  void AllocateIntermediateNodes() {
    auto &expr = *expr_;
    bool is_simple_expression = expr.GetSubexpressionCount() == 2 &&
                                expr[0].GetNodeType() != NodeType::Function &&
                                expr[1].GetNodeType() != NodeType::Function;
    DALI_ENFORCE(is_simple_expression, "Complex expression trees are not supported");
    // TODO(klecki): allocate memory for intermediate results and point the threads to them
  }

  std::unique_ptr<ExprNode> expr_;
  kernels::TensorListShape<> result_shape_;
  DALIDataType result_type_id_;
  std::vector<TileDesc> tile_cover_;
  std::vector<TileRange> tile_range_;
  std::vector<ExpressionImplTask> exec_order_;
  ExprImplCache cache_;
  int num_tasks_;
  static constexpr int kExtent = 1024;
  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_ARITHMETIC_ARITHMETIC_H_
