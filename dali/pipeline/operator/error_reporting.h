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
namespace dali {

// TODO(klecki): Throw this one into a namespace?

class DaliError : public std::runtime_error {
 public:
  // TODO(klecki): we have a place to control if we have additional metadata?
  explicit DaliError(const std::string &msg) : std::runtime_error(msg) {}

  void AddOriginInfo(const std::string &msg) {
    origin_info_ = msg + std::runtime_error::what();
  }

  const char* what() const noexcept override {
    return origin_info_.c_str();
  }
 private:
  std::string origin_info_;  // this is a bit wrong if we have bad alloc encountered
};

class DaliTypeError : public DaliError {
 public:
  explicit DaliTypeError(const std::string &msg) : DaliError(msg) {}


};

inline void ValidateInputType(DALIDataType actual, std::vector<DALIDataType> expected,
                              const std::string &additional_error) {
  if (std::find(expected.begin(), expected.end(), actual) == expected.end()) {
    throw DaliTypeError(make_string("Expected input type to be one of ",
                                    /* expected, */ ", but got: `", actual, "`. ",
                                    additional_error));
  }
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ERROR_REPORTING_H_
