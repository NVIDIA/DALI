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

#ifndef DALI_PIPELINE_OPERATORS_EXPRESSIONS_ARITHMETIC_H_
#define DALI_PIPELINE_OPERATORS_EXPRESSIONS_ARITHMETIC_H_

#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "dali/core/format.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/kernels/tensor_shape_print.h"
#include "dali/kernels/type_tag.h"
#include "dali/pipeline/operators/expressions/arithmetic_meta.h"
#include "dali/pipeline/operators/expressions/expression_impl_factory.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

using TileCover = std::tuple<std::vector<TileDesc>, std::vector<TileRange>>;

/**
 * @brief Divide the shape into groups of linear tiles
 */
inline TileCover GetTiledCover(const kernels::TensorListShape<> &shape, int tile_size,
                               int num_tiles_in_task) {
  Index total_elements = shape.num_elements();
  std::vector<TileDesc> descs;
  for (int sample_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
    int extent_idx = 0;
    Index sample_elements = shape[sample_idx].num_elements();
    for (Index covered = 0; covered < sample_elements; covered += tile_size, extent_idx++) {
      auto actual_tile_size =
          std::min(static_cast<Index>(tile_size), shape[sample_idx].num_elements() - covered);
      descs.push_back({sample_idx, extent_idx, static_cast<int>(actual_tile_size), tile_size});
    }
  }
  Index num_tasks = (descs.size() + num_tiles_in_task - 1) / num_tiles_in_task;
  std::vector<TileRange> ranges;
  ranges.reserve(num_tasks);
  for (int task = 0, tiles_used = 0; task < num_tasks; task++) {
    auto tiles_end = std::min(tiles_used + num_tiles_in_task, static_cast<int>(descs.size()));
    ranges.push_back({tiles_used, tiles_end});
    tiles_used = tiles_end;
  }
  return std::make_tuple(descs, ranges);
}


/**
 * @brief Recurse over expression tree and return the only matching layout
 */
template <typename Backend>
DLL_PUBLIC TensorLayout GetCommonLayout(ExprNode &expr, const workspace_t<Backend> &ws) {
  if (expr.GetNodeType() == NodeType::Constant) {
    return "";
  }
  if (expr.GetNodeType() == NodeType::Tensor) {
    auto &e = dynamic_cast<ExprTensor &>(expr);
    return ws.template InputRef<Backend>(e.GetInputIndex()).GetLayout();
  }
  if (expr.GetSubexpressionCount() == 0) {
    return "";
  }
  auto &func = dynamic_cast<ExprFunc&>(expr);
  auto result_layout = GetCommonLayout<Backend>(func[0], ws);
  for (int i = 1; i < expr.GetSubexpressionCount(); i++) {
    auto next_layout = GetCommonLayout<Backend>(func[i], ws);
    if (result_layout.empty()) {
      result_layout = next_layout;
      continue;
    }
    if (next_layout.empty()) {
      continue;
    }
    DALI_ENFORCE(
        result_layout == next_layout,
        make_string("Layouts of subexpressions", i - 1, "and", i, "for atihmetic operation",
                    func.GetFuncName(), "do not match. Expected", result_layout.c_str(), "got",
                    next_layout.c_str(), "."));
  }
  return result_layout;
}


/**
 * @brief Recurse over expression tree, fill the missing types of TensorInputs
 */
template <typename Backend>
DLL_PUBLIC DALIDataType PropagateTypes(ExprNode &expr, const workspace_t<Backend> &ws) {
  if (expr.GetNodeType() == NodeType::Constant) {
    return expr.GetTypeId();
  }
  if (expr.GetNodeType() == NodeType::Tensor) {
    auto &e = dynamic_cast<ExprTensor &>(expr);
    expr.SetTypeId(ws.template InputRef<Backend>(e.GetInputIndex()).type().id());
    return expr.GetTypeId();
  }
  if (expr.GetSubexpressionCount() == 2) {
    auto &func = dynamic_cast<ExprFunc&>(expr);
    auto left_type = PropagateTypes<Backend>(func[0], ws);
    auto right_type = PropagateTypes<Backend>(func[1], ws);
    func.SetTypeId(TypePromotion(left_type, right_type));
    return func.GetTypeId();
  }
  DALI_FAIL(make_string("Only binary expressions are supported. Got expression with",
                        expr.GetSubexpressionCount(), "subexpressions."));
}

struct ExprImplTask {
  ExprImplBase *impl;
  ExprImplContext ctx;
};

template <typename Backend>
inline void CreateExecutionOrder(std::vector<ExprImplTask> &order, const ExprNode &expr,
                                 ExprImplCache &cache) {
  if (expr.GetNodeType() != NodeType::Function) {
    return;
  }
  auto &func = dynamic_cast<const ExprFunc&>(expr);
  for (int i = 0; i < expr.GetSubexpressionCount(); i++) {
    CreateExecutionOrder<Backend>(order, func[i], cache);
  }
  order.push_back({cache.GetExprImpl<Backend>(func), {&func}});
}

template <typename Backend>
inline std::vector<ExprImplTask> CreateExecutionOrder(const ExprNode &expr,
                                                            ExprImplCache &cache) {
  std::vector<ExprImplTask> result;
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
  DALI_ENFORCE(left == right, "Input shapes of element-wise arithemtic operator \"" + op +
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
    auto &e = dynamic_cast<ExprTensor &>(expr);
    expr.SetShape(ws.template InputRef<Backend>(e.GetInputIndex()).shape());
    return expr.GetShape();
  }
  if (expr.GetSubexpressionCount() == 2) {
    auto &func = dynamic_cast<ExprFunc&>(expr);
    auto left_shape = PropagateShapes<Backend>(func[0], ws);
    auto right_shape = PropagateShapes<Backend>(func[1], ws);
    func.SetShape(ShapePromotion(func.GetFuncName(), left_shape, right_shape));
    return func.GetShape();
  }
  DALI_FAIL(make_string("Only binary expressions are supported. Got expression with",
                        expr.GetSubexpressionCount(), "subexpressions."));
}

/**
 * @brief Arithmetic operator capable of executing expression tree of element-wise
 *        arithmetic operations.
 *
 * Only expressions consisting of one function node with tensor inputs are now supported.
 *
 * There are 3 levels for unit of work.
 * - Thread (CPUBackend) or CUDA kernel invokation (GPUBackend)
 * - Task - group of tiles to process by thread or CUDA kernel
 * - Tile - describes a portion of linear buffer, we try to split the amount of work
 *          evenly into tasks.
 *
 * For CPUBackend we have fixed number of threads that get to process a number of tasks,
 * so the work is evenly distributed. For GPUBackend we pack all tiles into 1 task, to limit
 * the number of CUDA calls.
 */
template <typename Backend>
class ArithmeticGenericOp : public Operator<Backend> {
 public:
  inline explicit ArithmeticGenericOp(const OpSpec &spec) : Operator<Backend>(spec) {
    expr_ = ParseExpressionString(spec.GetArgument<std::string>("expression_desc"));
  }

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    output_desc.resize(1);

    if (!types_layout_inferenced_) {
      result_type_id_ = PropagateTypes<Backend>(*expr_, ws);
      result_layout_ = GetCommonLayout<Backend>(*expr_, ws);
      types_layout_inferenced_ = true;
    }

    result_shape_ = PropagateShapes<Backend>(*expr_, ws);
    AllocateIntermediateNodes();
    exec_order_ = CreateExecutionOrder<Backend>(*expr_, cache_);

    output_desc[0] = {result_shape_, TypeTable::GetTypeInfo(result_type_id_)};
    std::tie(tile_cover_, tile_range_) = GetTiledCover(result_shape_, kTileSize, kTaskSize);
    return true;
  }

  using Operator<Backend>::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override;

 private:
  void AllocateIntermediateNodes() {
    auto &expr = *expr_;
    bool is_simple_expression =
        expr.GetNodeType() == NodeType::Function && expr.GetSubexpressionCount() == 2;
    auto &func = dynamic_cast<ExprFunc&>(expr);
    for (int i = 0; i < func.GetSubexpressionCount(); i++) {
      is_simple_expression = is_simple_expression && func[i].GetNodeType() != NodeType::Function;
    }

    DALI_ENFORCE(is_simple_expression,
                 "Complex expression trees are not yet supported. Only expressions containing one "
                 "binary function node are supported.");
    // TODO(klecki): allocate memory for intermediate results and point the threads to them
  }

  std::unique_ptr<ExprNode> expr_;
  kernels::TensorListShape<> result_shape_;
  bool types_layout_inferenced_ = false;
  DALIDataType result_type_id_;
  TensorLayout result_layout_;
  std::vector<TileDesc> tile_cover_;
  std::vector<TileRange> tile_range_;
  std::vector<ExprImplTask> exec_order_;
  ExprImplCache cache_;
  // For CPU we limit the tile size to limit the sizes of intermediate buffers
  // For GPU it's better to execute more at one time.
  static constexpr int kTileSize = std::is_same<Backend, CPUBackend>::value ? 4096 : 16384;
  // CPU packs up to 64 tiles in one task, GPU porcesses all of them in one task
  static constexpr int kTaskSize =
      std::is_same<Backend, CPUBackend>::value ? 64 : std::numeric_limits<int>::max();
  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_EXPRESSIONS_ARITHMETIC_H_
