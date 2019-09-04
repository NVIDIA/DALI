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
// All python headers must be included before std headers due to macro redefinition error
#include <vector>  // NOLINT
#include "dali/pipeline/operators/python_function/util/copy_with_stride.h"

namespace dali {

namespace detail {

template <typename Backend>
constexpr DLDeviceType Backend2DLDevice() {
  if (std::is_same<Backend, CPUBackend>::value) {
    return kDLCPU;
  } else {
    return kDLGPU;
  }
}

template <typename Backend>
std::vector<DLMTensorPtr> CastToDLTensorList(py::list &list, Index exp_size, Index out_idx) {
  DALI_ENFORCE(list.size() == static_cast<size_t>(exp_size),
      "Function called by DLTensorPythonFunction returned tensor list of wrong size at idx "
      + std::to_string(out_idx) + ". Returned list is of a size "
      + std::to_string(exp_size) + " and should be of size " + std::to_string(exp_size));
  std::vector<DLMTensorPtr> result;
  result.reserve(exp_size);
  if (exp_size) {
    DALI_ENFORCE(py::capsule::check_(list[0]),
                 "Function called by DLTensorPythonFunction "
                 "should return a list of DLPack tensors.");
    auto caps = py::cast<py::capsule>(list[0]);
    result.push_back(DLMTensorPtrFromCapsule(caps));
    DALI_ENFORCE(result[0]->dl_tensor.ctx.device_type == Backend2DLDevice<Backend>(),
        "Wrong output backend");
    auto dtype = result[0]->dl_tensor.dtype;
    for (Index i = 1; i < exp_size; ++i) {
      auto caps = py::cast<py::capsule>(list[i]);
      result.push_back(DLMTensorPtrFromCapsule(caps));
      DALI_ENFORCE(result[i]->dl_tensor.ctx.device_type == Backend2DLDevice<Backend>(),
                   "Wrong output backend.");
      DALI_ENFORCE(DLToDALIType(result[i]->dl_tensor.dtype) == DLToDALIType(dtype),
          "Output DLPack tensor list should have consistent data type.");
    }
  }
  return result;
}

template <typename Backend>
void CopyDlTensor(void *out_data, DLMTensorPtr &dlm_tensor_ptr) {
  auto &dl_tensor = dlm_tensor_ptr->dl_tensor;
  auto item_size = dl_tensor.dtype.bits / 8;
  std::vector<Index> strides(dl_tensor.ndim);
  if (dl_tensor.strides) {
    for (Index i = 0; i < dl_tensor.ndim; ++i) strides[i] = dl_tensor.strides[i] * item_size;
  } else {
    strides.back() = item_size;
    for (Index i = dl_tensor.ndim - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dl_tensor.shape[i + 1];
    }
  }
  CopyWithStride<Backend>(out_data, dl_tensor.data, strides.data(),
                          dl_tensor.shape, dl_tensor.ndim, item_size);
}

template <typename Backend>
py::list PrepareInputs(workspace_t<Backend> &ws);

template <typename Backend>
void CopyOutputs(workspace_t<Backend> &ws, py::tuple &output);

}  // namespace detail

template <typename Backend>
class DLTensorPythonFunctionImpl : public PythonFunctionImplBase<Backend> {
 public:
  inline explicit DLTensorPythonFunctionImpl(const OpSpec &spec)
    : PythonFunctionImplBase<Backend>(spec) {}

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    std::lock_guard<std::mutex> operator_guard(operator_lock);
    py::gil_scoped_acquire interpreter_guard{};
    py::object output_o;
    try {
      output_o = python_function(*py::tuple(detail::PrepareInputs<Backend>(ws)));
    } catch(const py::error_already_set &e) {
      throw std::runtime_error(to_string("DLTensorPythonFunction error: ") + to_string(e.what()));
    }
    if (!output_o.is_none()) {
      py::tuple output = (py::tuple::check_(output_o)) ? output_o : py::make_tuple(output_o);
      detail::CopyOutputs<Backend>(ws, output);
    } else {
      DALI_ENFORCE(ws.NumOutput() == 0, "Python function returned 0 outputs and "
          + std::to_string(ws.NumOutput()) + " were expected.");
    }
  };

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
  using PythonFunctionImplBase<Backend>::python_function;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_PYTHON_FUNCTION_DLTENSOR_FUNCTION_H_
