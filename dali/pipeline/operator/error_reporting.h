// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_ERROR_REPORTING_H_
#define DALI_PIPELINE_OPERATOR_ERROR_REPORTING_H_

#include <exception>
#include <stdexcept>

// #include "dali/core/error_handling.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/op_spec.h"
namespace dali {

// TODO(klecki): Throw this one into a namespace?

struct DaliFrameSummary {
  DaliFrameSummary(std::string &&filename, int lineno, std::string &&name, std::string &&line) :
    filename(std::move(filename)), lineno(lineno), name(std::move(name)), line(std::move(line)) {}
  std::string filename;
  int lineno;
  std::string name;
  std::string line;
};

/**
 * @brief Get the origin stack trace for operator constructed with given spec.
 * The stack trace defines frames between the invocation od pipeline definition and operator call.
 * The returned DaliFrameSummary corresponds to Python traceback.FrameSummary
 * The `line` context may be invalid in autograph transformed code.
 */
std::vector<DaliFrameSummary> GetOperatorOriginInfo(const OpSpec &spec);

std::string FormatStack(const std::vector<DaliFrameSummary> &stack_summary, bool include_context);

class DaliError : public std::exception {
  /**
   * @brief Error class for exceptions produced in the DALI pipeline. Subtypes should be used
   * for specific errors that will be mapped into appropriate Python error types of matching names.
   *
   * When DALI executor sees this error it will add origin information to the message and rethrow.
   */
 public:
  // TODO(klecki): Do we want to mark when we have the origin info or introduce placeholders
  // for op name, etc?
  explicit DaliError(const std::string &msg) : std::exception(), msg_(msg) {}

  void AddOriginInfo(const std::string &msg) {
    msg_ = msg + msg_;
  }

  const char* what() const noexcept override {
    return msg_.c_str();
  }

 private:
  std::string msg_;
};

class DaliTypeError : public DaliError {
  /**
   * @brief Error class that will be mapped to Python TypeError
   */
 public:
  explicit DaliTypeError(const std::string &msg) : DaliError(msg) {}
};


class DaliValueError : public DaliError {
  /**
   * @brief Error class that will be mapped to Python ValueError
   */
 public:
  explicit DaliValueError(const std::string &msg) : DaliError(msg) {}
};

// TODO(klecki): Consider optional input index.
// TODO(klecki): Consider (argument) input name!
inline void ValidateInputType(DALIDataType actual, DALIDataType expected,
                              const std::string &additional_error = "") {
  if (actual != expected) {
    throw DaliTypeError(make_string("Expected input type to be `",
                                    expected, "`, but got: `", actual, "`. ",
                                    additional_error));
  }
}

inline void ValidateInputType(DALIDataType actual, std::vector<DALIDataType> expected,
                              const std::string &additional_error) {
  if (std::find(expected.begin(), expected.end(), actual) == expected.end()) {
    throw DaliTypeError(make_string("Expected input type to be one of ",
                                    /* expected, */ ", but got: `", actual, "`. ",
                                    additional_error));
  }
}

inline void ValidateInputDim(int actual, int expected,
                              const std::string &additional_error = "") {
  if (actual != expected) {
    throw DaliValueError(make_string("Expected input dim to be `",
                                    expected, "`, but got: `", actual, "`. ",
                                    additional_error));
  }
}

inline void ValidateInputDim(int actual, std::vector<int> expected,
                              const std::string &additional_error) {
  if (std::find(expected.begin(), expected.end(), actual) == expected.end()) {
    throw DaliValueError(make_string("Expected input dim to be one of ",
                                    /* expected, */ ", but got: `", actual, "`. ",
                                    additional_error));
  }
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ERROR_REPORTING_H_
