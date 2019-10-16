// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_COMMON_H_
#define DALI_PIPELINE_OPERATOR_COMMON_H_

#include <vector>
#include <string>

#include "dali/pipeline/operator/op_spec.h"

namespace dali {
template <typename T>
inline void GetSingleOrRepeatedArg(const OpSpec &spec, vector<T> &result,
                                   const std::string &argName, size_t repeat_count = 2) {
  if (!spec.TryGetRepeatedArgument<T>(result, argName)) {
      T scalar = spec.GetArgument<T>(argName);
      result.assign(repeat_count, scalar);
  } else if (result.size() == 1 && repeat_count != 1) {
      T scalar = result.front();
      result.assign(repeat_count, scalar);
  }

  DALI_ENFORCE(result.size() == repeat_count,
      "Argument \"" + argName + "\" expects either a single value "
      "or a list of " + to_string(repeat_count) + " elements. " +
      to_string(result.size()) + " given.");
}

}  // namespace dali
#endif  // DALI_PIPELINE_OPERATOR_COMMON_H_
