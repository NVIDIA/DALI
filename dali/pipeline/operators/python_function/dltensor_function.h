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

#ifndef DALI_PIPELINE_OPERATORS_PYTHON_FUNCTION_DLTENSOR_FUNCTION_H_
#define DALI_PIPELINE_OPERATORS_PYTHON_FUNCTION_DLTENSOR_FUNCTION_H_

#include "dali/pipeline/operators/python_function/python_function.h"

namespace dali {

template <typename Backend>
class DLTensorPythonFunctionImpl : public PythonFunctionImplBase<Backend> {
 public:
  inline explicit DLTensorPythonFunctionImpl(const OpSpec &spec)
    : PythonFunctionImplBase<Backend>(spec) {}

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(workspace_t<Backend> *ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_PYTHON_FUNCTION_DLTENSOR_FUNCTION_H_
