// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_OPERATORS_EXCEPTION_H_
#define DALI_TEST_OPERATORS_EXCEPTION_H_

#include <string>
#include <vector>

#include "dali/pipeline/operator/error_reporting.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class ThrowExceptionOp : public Operator<Backend> {
 public:
  inline explicit ThrowExceptionOp(const OpSpec &spec) : Operator<Backend>(spec) {
    if (spec.GetArgument<bool>("constructor")) {
      throw DaliRuntimeError("Error in constructor");
    }
  }

  inline ~ThrowExceptionOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(ThrowExceptionOp);
  USE_OPERATOR_MEMBERS();

 protected:
  bool HasContiguousOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {
    auto message = spec_.template GetArgument<std::string>("message");
    auto exception_type = spec_.template GetArgument<std::string>("exception_type");

    if (exception_type == "RuntimeError") {
      throw DaliRuntimeError(message);
    } else if (exception_type == "IndexError") {
      throw DaliIndexError(message);
    } else if (exception_type == "TypeError") {
      throw DaliTypeError(message);
    } else if (exception_type == "ValueError") {
      throw DaliValueError(message);
    } else if (exception_type == "StopIteration") {
      throw DaliStopIteration(message);
    } else if (exception_type == "std::invalid_argument") {
      throw std::invalid_argument(message);
    } else if (exception_type == "std::domain_error") {
      throw std::domain_error(message);
    } else if (exception_type == "std::length_error") {
      throw std::length_error(message);
    } else if (exception_type == "std::out_of_range") {
      throw std::out_of_range(message);
    } else if (exception_type == "std::range_error") {
      throw std::range_error(message);
    } else if (exception_type == "std::runtime_error") {
      throw std::runtime_error(message);
    } else if (exception_type == "std::string") {
      throw message;
    }
    throw DaliError("Unknown error kind.");
  }
};

}  // namespace dali

#endif  // DALI_TEST_OPERATORS_EXCEPTION_H_
