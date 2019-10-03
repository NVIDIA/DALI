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

#ifndef DALI_PIPELINE_OPERATORS_ARITHMETIC_EXPRESSION_TREE_H_
#define DALI_PIPELINE_OPERATORS_ARITHMETIC_EXPRESSION_TREE_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/core/any.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operators/arithmetic/arithmetic_meta.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

struct TileDesc {
  int sample_idx;
  int extent_idx;  // extents are number inside given sample_idx
  int task_idx;  // thread assgined to this extent
  int64_t extent_size;
};

struct TileRange {
  int begin;
  int end;
};

inline std::ostream& operator<<(std::ostream& os, const TileDesc& v) {
  os << "{ " << v.sample_idx << ", " << v.extent_idx << ", " << v.task_idx << ", " << v.extent_size
     << "}";
  return os;
}

class ExprNode;

struct ExpressionImplContext {
  const ExprNode *node;
};

/**
 * @brief Part responsible for loop-execution over tile, the implementation knows the types
 * and can access the inputs internally
 */
class ExpressionImplBase {
 public:
  virtual void Execute(ArgumentWorkspace &workspace, const OpSpec &spec, ExpressionImplContext &ctx,
                       const std::vector<TileDesc> &tiles, TileRange range) = 0;
  virtual ~ExpressionImplBase() = default;
};

enum class NodeType {
  Function,
  Constant,
  Tensor
};

inline std::string GetAbbreviation(NodeType t) {
  switch (t) {
    case NodeType::Function:
      return "F";
    case NodeType::Constant:
      return "C";
    case NodeType::Tensor:
      return "T";
    default:
      DALI_FAIL("Unrecognized NodeType.");
  }
}

/**
 * @brief base class for nodes of ExpressionTree
 *
 * TODO(klecki): Using unique_ptrs for ownership, we may generalize it to DAG, that would require
 * some changes
 */
class ExprNode {
 public:
  virtual const std::string &GetOp() const {
    DALI_FAIL("No op in this expression node.");
  }

  virtual std::string GetNodeDesc() const {
    return GetOutputDesc();
  }

  virtual std::string GetOutputDesc() const {
    auto op_type = TypeTable::GetTypeInfo(GetTypeId()).name();
    std::string result = GetAbbreviation(GetNodeType());
    result += IsScalarLike(GetShape()) ? "C:" : "T:";
    return result + op_type;
  }

  virtual NodeType GetNodeType() const = 0;

  std::unique_ptr<ExpressionImplBase> impl;

  void SetTypeId(DALIDataType type_id) {
    type_id_ = type_id;
  }

  DALIDataType GetTypeId() const {
    return type_id_;
  }

  void SetShape(const kernels::TensorListShape<> &shape) {
    shape_ = shape;
  }

  const kernels::TensorListShape<> &GetShape() const {
    return shape_;
  }

  virtual int GetSubexpressionCount() const {
    return 0;
  }

  virtual ExprNode &operator[](int i) {
    DALI_FAIL("No subexpressions for this node.");
  }

  virtual const ExprNode &operator[](int i) const {
    DALI_FAIL("No subexpressions for this node.");
  }

  virtual ~ExprNode() = default;

 private:
  DALIDataType type_id_ = DALI_NO_TYPE;
  kernels::TensorListShape<> shape_ = {};
};

class ExprFunc : public ExprNode {
 public:
  explicit ExprFunc(const std::string &op) : op_(op) {}

  ExprFunc(const std::string &op, int num_input) : op_(op) {
    subexpr_.resize(num_input);
  }

  const std::string &GetOp() const override {
    return op_;
  }

  std::string GetNodeDesc() const override {
    auto op_type = TypeTable::GetTypeInfo(GetTypeId()).name();
    std::string result = op_ + (IsScalarLike(GetShape()) ? ":C:" : ":T:") + op_type + "(";
    for (int i = 0; i < GetSubexpressionCount(); i++) {
      result += (*this)[i].GetOutputDesc();
      if (i < GetSubexpressionCount() - 1) {
        result += " ";
      }
    }
    result += ")";
    return result;
  }

  NodeType GetNodeType() const override {
    return NodeType::Function;
  }

  int GetSubexpressionCount() const override {
    return subexpr_.size();
  }

  void AddSubexpression(std::unique_ptr<ExprNode> expr) {
    subexpr_.push_back(std::move(expr));
  }

  ExprNode &operator[](int i) override {
    return *subexpr_[i];
  }

  const ExprNode &operator[](int i) const override {
    return *subexpr_[i];
  }

 private:
  std::string op_;
  std::vector<std::unique_ptr<ExprNode>> subexpr_;
};

class ExprTensor : public ExprNode {
 public:
  explicit ExprTensor(int mapped_input) : mapped_input_(mapped_input) {}

  NodeType GetNodeType() const override {
    return NodeType::Tensor;
  }

  int GetMappedInput() const {
    return mapped_input_;
  }

 private:
  int mapped_input_ = -1;
};

class ExprConstant : public ExprNode {
 public:
  ExprConstant(int scalar_id, DALIDataType type_id) : mapped_input_(scalar_id) {
    SetTypeId(type_id);
  }

  NodeType GetNodeType() const override {
    return NodeType::Constant;
  }

  int GetMappedInput() const {
    return mapped_input_;
  }

 private:
  int mapped_input_ = -1;
};



DLL_PUBLIC std::unique_ptr<ExprNode> ParseExpressionString(const std::string &expr);

}  // namespace dali


#endif  // DALI_PIPELINE_OPERATORS_ARITHMETIC_EXPRESSION_TREE_H_
