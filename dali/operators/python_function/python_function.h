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

#ifndef DALI_OPERATORS_PYTHON_FUNCTION_PYTHON_FUNCTION_H_
#define DALI_OPERATORS_PYTHON_FUNCTION_PYTHON_FUNCTION_H_

#include <dali/util/pybind.h>
#include <pybind11/embed.h>
#include <vector>
#include "dali/pipeline/operator/operator.h"

namespace dali {

extern std::mutex operator_lock;

template <typename Backend>
class PythonFunctionImplBase : public Operator<Backend> {
 public:
  explicit PythonFunctionImplBase(const OpSpec &spec)
  : Operator<Backend>(spec)
  , python_function(py::reinterpret_borrow<py::object>(
      reinterpret_cast<PyObject*>(spec.GetArgument<int64_t>("function_id")))) {}

 protected:
  py::object python_function;
};

template <typename Backend>
class PythonFunctionImpl : public PythonFunctionImplBase<Backend> {
 public:
  inline explicit PythonFunctionImpl(const OpSpec &spec)
    : PythonFunctionImplBase<Backend>(spec) {}

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(Workspace<Backend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

cudaStream_t GetCurrentStream();

void SetCurrentStream(cudaStream_t stream);

}  // namespace dali

#endif  // DALI_OPERATORS_PYTHON_FUNCTION_PYTHON_FUNCTION_H_
