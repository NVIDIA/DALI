// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "dali/operators/conditional/validation.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali {

void EnforceConditionalInputKind(const TensorList<CPUBackend> &input, const std::string &name,
                                 const std::string &side, bool enforce_type){
  std::string preamble =
      make_string("Logical expression ``", name, "`` is restricted to scalar (0-d tensors) inputs",
                  enforce_type ? " of bool type." : ".");

  std::string suggestion =
      "\n\nThis input restriction allows the logical expressions to always return scalar boolean "
      "outputs and to be used in unambiguous way in DALI conditionals. You may use bitwise "
      "arithmetic operators ``&``, ``|`` if you need to process inputs of higher dimensionality or "
      "different type - those operations performed on boolean inputs are equivalent to logical "
      "expressions.";

  std::string in_side_mention = side.empty() ? "." : make_string(" on the ", side, ".");

  auto dim = input.shape().sample_dim();
  DALI_ENFORCE(dim == 0,
               make_string(preamble, " Got a ", dim, "-d input", in_side_mention, suggestion));

  if (enforce_type) {
    auto type = input.type();
    DALI_ENFORCE(type == DALI_BOOL,
                make_string(preamble, " Got an input of type ", type, in_side_mention, suggestion));
  }
}

}