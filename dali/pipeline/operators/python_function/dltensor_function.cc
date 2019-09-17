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

#include "dali/pipeline/operators/python_function/dltensor_function.h"
#include "dali/pipeline/operators/python_function/util/copy_with_stride.h"

namespace dali {

DALI_SCHEMA(DLTensorPythonFunctionImpl)
    .AddParent("PythonFunctionImplBase")
    .NumInput(0, 256)
    .OutputFn([](const OpSpec &spec) {return spec.GetArgument<int>("num_outputs");})
    .MakeInternal();

DALI_SCHEMA(DLTensorPythonFunction)
    .AddParent("PythonFunctionBase")
    .DocStr("Execute a python function that operates on DLPack tensors.")
    .NumInput(0, 256)
    .NoPrune();

namespace detail {

template <>
py::list PrepareDLTensorInputs<CPUBackend>(HostWorkspace &ws) {
  py::list input_tuple;
  for (Index idx = 0; idx < ws.NumInput(); ++idx) {
    py::list dl_tensor_list;
    for (Index i = 0; i < ws.NumInputAtIdx(idx); ++i) {
      auto &t = ws.Input<CPUBackend>(idx, i);
      auto dl_capsule = TensorToDLPackView(const_cast<Tensor<CPUBackend>&>(t));
      dl_tensor_list.append(dl_capsule);
    }
    input_tuple.append(dl_tensor_list);
  }
  return input_tuple;
}

template <>
py::list PrepareDLTensorInputs<GPUBackend>(DeviceWorkspace &ws) {
  py::list input_tuple;
  for (Index idx = 0; idx < ws.NumInput(); ++idx) {
    auto &tlist = ws.InputRef<GPUBackend>(idx);
    py::list dl_tensor_list = TensorListToDLPackView(tlist);
    input_tuple.append(dl_tensor_list);
  }
  return input_tuple;
}

kernels::TensorListShape<> GetDLTensorListShape(const std::vector<DLMTensorPtr>& dl_tensors) {
  kernels::TensorListShape<> list_shape{};
  list_shape.resize(dl_tensors.size(), dl_tensors[0]->dl_tensor.ndim);
  for (size_t i = 0; i < dl_tensors.size(); ++i) {
    auto &dl_tensor = dl_tensors[i]->dl_tensor;
    assert(dl_tensor.ndim == list_shape.sample_dim());
    list_shape.set_tensor_shape(i, make_span(dl_tensor.shape, dl_tensor.ndim));
  }
  return list_shape;
}

template <>
void CopyDLTensorOutputs<CPUBackend>(HostWorkspace &ws, py::tuple &return_tuple, int batch_size) {
  for (Index idx = 0; idx < ws.NumOutput(); ++idx) {
    py::list dl_list = py::cast<py::list>(return_tuple[idx]);
    auto dl_tensors = CastToDLTensorList<CPUBackend>(dl_list, batch_size, idx);
    if (dl_tensors.empty()) continue;
    auto &tvec = ws.OutputRef<CPUBackend>(idx);
    tvec.set_type(TypeTable::GetTypeInfo(DLToDALIType(dl_tensors[0]->dl_tensor.dtype)));
    tvec.Resize(GetDLTensorListShape(dl_tensors));
    auto &thread_pool = ws.GetThreadPool();
    for (int i = 0; i < batch_size; ++i) {
      thread_pool.DoWorkWithID([&, i](int) {
        CopyDlTensor<CPUBackend>(tvec[i].raw_mutable_data(), dl_tensors[i]);
      });
    }
    thread_pool.WaitForWork();
  }
}

template <>
void CopyDLTensorOutputs<GPUBackend>(DeviceWorkspace &ws, py::tuple &return_tuple, int batch_size) {
  for (Index idx = 0; idx < ws.NumOutput(); ++idx) {
    py::list dl_list = py::cast<py::list>(return_tuple[idx]);
    auto dl_tensors = CastToDLTensorList<GPUBackend>(dl_list, batch_size, idx);
    if (dl_tensors.empty()) continue;
    auto &tlist = ws.OutputRef<GPUBackend>(idx);
    tlist.set_type(TypeTable::GetTypeInfo(DLToDALIType(dl_tensors[0]->dl_tensor.dtype)));
    tlist.Resize(GetDLTensorListShape(dl_tensors));
    for (int i = 0; i < batch_size; ++i) {
      CopyDlTensor<GPUBackend>(tlist.raw_mutable_tensor(i), dl_tensors[i], ws.stream());
    }
  }
}

}  // namespace detail

DALI_REGISTER_OPERATOR(DLTensorPythonFunctionImpl, DLTensorPythonFunctionImpl<CPUBackend>, CPU);

DALI_REGISTER_OPERATOR(DLTensorPythonFunctionImpl, DLTensorPythonFunctionImpl<GPUBackend>, GPU);

}  // namespace dali
