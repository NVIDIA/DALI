// Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>

#include "dali/kernels/type_tag.h"
#include "dali/operators/math/expressions/arithmetic.h"

namespace dali {
namespace expr {

std::optional<DALIDataType> PropagateTypes(
      ExprNode &expr, span<const std::optional<DALIDataType>> input_types) {
  if (expr.GetNodeType() == NodeType::Constant) {
    return expr.GetTypeId();
  }
  if (expr.GetNodeType() == NodeType::Tensor) {
    auto &e = dynamic_cast<ExprTensor &>(expr);
    int idx = e.GetInputIndex();
    if (idx < 0)
      throw std::out_of_range("Negative input index encountered.");

    if (idx >= input_types.size()) {
      throw std::out_of_range(make_string(
        "Input index ", idx, " is out of range. "
        "Only ", input_types.size(), " inputs are present."));
    }
    if (!input_types[idx])
      return std::nullopt;

    expr.SetTypeId(*input_types[e.GetInputIndex()]);
    return expr.GetTypeId();
  }
  auto &func = dynamic_cast<ExprFunc &>(expr);
  int subexpression_count = func.GetSubexpressionCount();
  DALI_ENFORCE(0 < subexpression_count && subexpression_count <= kMaxArity,
               "Only unary, binary and ternary expressions are supported");

  SmallVector<DALIDataType, kMaxArity> types;
  types.resize(subexpression_count);
  for (int i = 0; i < subexpression_count; i++) {
    auto subexpr_type = PropagateTypes(func[i], input_types);
    if (!subexpr_type)
      return std::nullopt;
    types[i] = *subexpr_type;
  }
  expr.SetTypeId(TypePromotion(NameToOp(func.GetFuncName()), make_span(types)));
  return expr.GetTypeId();
}


DALIDataType PropagateTypes(ExprNode &expr, const Workspace &ws) {
  SmallVector<std::optional<DALIDataType>, 8> input_types;
  for (int i = 0; i < ws.NumInput(); i++)
    input_types.push_back(ws.GetInputDataType(i));
  return PropagateTypes(expr, make_cspan(input_types)).value();
}

std::optional<DALIDataType> PropagateTypes(ExprNode &expr, const OpSpec &spec) {
  SmallVector<std::optional<DALIDataType>, 8> input_types;
  for (int i = 0; i < spec.NumInput(); i++)
    input_types.push_back(spec.InputDesc(i).dtype);
  return PropagateTypes(expr, make_cspan(input_types));
}


template <>
void ArithmeticGenericOp<CPUBackend>::RunImpl(Workspace &ws) {
  PrepareSamplesPerTask<CPUBackend>(samples_per_task_, exec_order_, ws, constant_storage_, spec_);
  auto &pool = ws.GetThreadPool();
  ws.Output<CPUBackend>(0).SetLayout(result_layout_);

  int ndim = 1;
  for (const auto &samples : samples_per_task_) {
    for (const auto &sample : samples) {
      ndim = std::max(ndim, sample.output.shape.sample_dim());
    }
  }
  if (ndim == 1) {
    std::tie(tile_cover_, tile_range_) = GetTiledCover(result_shape_, kTileSize, kTaskSize);
  } else {
    std::tie(tile_cover_, tile_range_) = GetOneTilePerSample(result_shape_);
  }

  int batch_size = ws.GetInputBatchSize(0);
  for (size_t task_idx = 0; task_idx < tile_range_.size(); task_idx++) {
    pool.AddWork(
        [&, task_idx](int thread_idx) {
          auto range = tile_range_[task_idx];
          // Go over "tiles"
          for (int extent_idx = range.begin; extent_idx < range.end; extent_idx++) {
            // Go over expression tree in some provided order
            for (size_t i = 0; i < exec_order_.size(); i++) {
              assert(batch_size == static_cast<int>(samples_per_task_[i].size()));
              auto samples = make_cspan(samples_per_task_[i]);
              exec_order_[i].impl->Execute(exec_order_[i].ctx, samples,
                                           make_cspan(&tile_cover_[extent_idx], 1));
            }
          }
        },
        -task_idx);  // FIFO order, since the work is already divided to similarly sized chunks
  }
  pool.RunAll();
}

}  // namespace expr

DALI_SCHEMA(_ArithmeticGenericOp)
    .DocStr(R"code(Arithmetic operator capable of executing expression tree of element-wise
arithmetic operations.)code")
    .AddArg("expression_desc", R"code(Polish notation describing the expression extendend with
parentheses, see https://en.wikipedia.org/wiki/Polish_notation.
Functions and operations are indicated by names, followed by parentheses.
Inputs (subexpressions) are placed in the parentheses and are separated by spaces,
&<uint> indicates tensor input, $<uint>:<type_string> indicates constant.

More formally using BNF grammar::

  <expr>    ::= <call> | <scalar> | <input>
  <subexpr> ::= <expr> | <expr> " " <subexpr>
  <call>    ::= <name> "(" <subexpr> ")"
  <name>    ::= identifier starting with alphabetic character
  <input>   ::= "&" <uint>
  <scalar>  ::= "$" <uint> ":" <type>
  <uint>    ::= unsigned integer
  <type>    ::= uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64
                      | float16 | float32 | float64

Examples::

  add(&0 mul(&1 $0:int8))
  add(&0 rand()))code",
            DALIDataType::DALI_STRING, false)
    .AddOptionalArg<std::vector<int32_t>>("integer_constants", "", nullptr, true)
    .NumInput(1, 64)  // Some arbitrary number that needs to be validated in operator
    .AddOptionalArg<std::vector<float>>("real_constants", "", nullptr, true)
    .NumOutput(1)
    .MakeDocHidden()
    .OutputNDim(0, [](const OpSpec &spec)->std::optional<int> {
      int ndim = 0;
      for (int i = 0; i < spec.NumRegularInput(); i++) {
        auto &desc = spec.InputDesc(i);
        if (!desc.ndim)
          return std::nullopt;
        if (*desc.ndim > ndim)
          ndim = *desc.ndim;
      }
      return ndim;
    })
    .OutputDType(0, [](const OpSpec &spec)->std::optional<DALIDataType> {
      try {
        auto ex = expr::ParseExpressionString(spec.GetArgument<std::string>("expression_desc"));
        if (!ex)
          return std::nullopt;
        return PropagateTypes(*ex, spec);
      } catch (const std::exception &) {
        return std::nullopt;
      }
    })
    .OutputLayout(0, [](const OpSpec &spec)->std::optional<TensorLayout> {
      // Layouts must match or be suffixes (when broadcasting), e.g.:
      // HWC + WC
      // If any layout is not known, bail out.
      // Empty layout is considered a suffix and skipped.
      std::optional<TensorLayout> layout;
      int ndim = 0;
      for (int i = 0; i < spec.NumRegularInput(); i++) {
        auto &desc = spec.InputDesc(i);
        if (!desc.layout)
          return std::nullopt;
        if (!desc.ndim)
          return std::nullopt;
        if (*desc.ndim > ndim)
          ndim = *desc.ndim;
        if (layout.has_value()) {
          if (layout->ndim() > desc.layout->ndim()) {
            if (layout->sub(layout->ndim() - desc.layout->ndim()) != *desc.layout)
              return std::nullopt;
          } else {
            if (desc.layout->sub(desc.layout->ndim() - layout->ndim()) != *layout)
              return std::nullopt;
            layout = desc.layout;
          }
        } else {
          layout = desc.layout;
        }
      }
      if (layout && layout->ndim() == ndim)
        return layout;
      else
        return std::nullopt;
    });

DALI_REGISTER_OPERATOR(_ArithmeticGenericOp, expr::ArithmeticGenericOp<CPUBackend>, CPU);

}  // namespace dali
