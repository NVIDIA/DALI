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

#include <vector>

#include "dali/kernels/type_tag.h"
#include "dali/operators/math/expressions/arithmetic.h"

namespace dali {

template <>
void ArithmeticGenericOp<CPUBackend>::RunImpl(HostWorkspace &ws) {
  PrepareTilesForTasks<CPUBackend>(tiles_per_task_, exec_order_, tile_cover_, ws, constant_storage_,
                                   spec_);
  auto &pool = ws.GetThreadPool();
  ws.OutputRef<CPUBackend>(0).SetLayout(result_layout_);
  for (size_t task_idx = 0; task_idx < tile_range_.size(); task_idx++) {
    pool.AddWork([this, task_idx](int thread_idx) {
      auto range = tile_range_[task_idx];
      // Go over "tiles"
      for (int extent_idx = range.begin; extent_idx < range.end; extent_idx++) {
        // Go over expression tree in some provided order
        for (size_t i = 0; i < exec_order_.size(); i++) {
          exec_order_[i].impl->Execute(exec_order_[i].ctx, tiles_per_task_[i],
                                       {extent_idx, extent_idx + 1});
        }
      }
    }, -task_idx);  // FIFO order, since the work is already divided to similarly sized chunks
  }
  pool.RunAll();
}

DALI_SCHEMA(ArithmeticGenericOp)
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
    .AddOptionalArg("integer_constants", "", std::vector<int32_t>{}, true)
    .NumInput(1, 64)  // Some arbitrary number that needs to be validated in operator
    .AddOptionalArg("real_constants", "", std::vector<float>{}, true)
    .NumOutput(1)
    .MakeDocHidden();

DALI_REGISTER_OPERATOR(ArithmeticGenericOp, ArithmeticGenericOp<CPUBackend>, CPU);

}  // namespace dali
