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

#include <string>
#include <utility>
#include <vector>

#include "dali/core/api_helper.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/op_spec.h"
namespace dali {

// TODO(klecki): Throw this one into a namespace?

/**
 * @brief Direct equivalent of Python's traceback.FrameSummary:
 * https://docs.python.org/3/library/traceback.html#traceback.FrameSummary
 *
 */
struct DaliFrameSummary {
  DaliFrameSummary(std::string &&filename, int lineno, std::string &&name, std::string &&line)
      : filename(std::move(filename)),
        lineno(lineno),
        name(std::move(name)),
        line(std::move(line)) {}
  std::string filename;
  int lineno;
  std::string name;
  std::string line;
};

/**
 * @brief Get the origin stack trace for operator constructed with given spec.
 * The stack trace defines frames between invocation of the pipeline definition and the operator
 * call. The returned DaliFrameSummary corresponds to Python traceback.FrameSummary, but the `line`
 * context may be invalid in some autograph transformed code.
 */
DLL_PUBLIC std::vector<DaliFrameSummary> GetOperatorOriginInfo(const OpSpec &spec);

DLL_PUBLIC std::string FormatStack(const std::vector<DaliFrameSummary> &stack_summary,
                                   bool include_context);

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ERROR_REPORTING_H_
