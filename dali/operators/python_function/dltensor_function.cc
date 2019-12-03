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

#include "dali/operators/python_function/dltensor_function.h"
#include "dali/pipeline/util/copy_with_stride.h"

namespace dali {

DALI_SCHEMA(DLTensorPythonFunctionImpl)
    .AddOptionalArg("synchronize_stream", "Synchronize CUDA stream", true)
    .NumInput(0, 256)
    .OutputFn([](const OpSpec &spec) {return spec.GetArgument<int>("num_outputs");})
    .MakeInternal()
    .AddParent("PythonFunctionImplBase");

DALI_SCHEMA(DLTensorPythonFunction)
    .DocStr(R"code(Execute a python function that operates on DLPack tensors.
In case of the GPU operator it is a user's responsibility to synchronize the device code with DALI.
This can be accomplished by synchronizing DALI's work before the operator call
with the `synchronize_stream` flag (true by default) and then making sure
the scheduled device tasks are finished within the operator call.
Alternatively, the gpu code can be done on the DALI's stream
which may be determined by calling the `current_dali_stream()` function.
In this case, the `synchronize_stream` flag can be set to false.')code")
    .AddOptionalArg("synchronize_stream",
        R"code(Make DALI synchronize its CUDA stream before calling the python function.
Should be set to false only if the called function schedules the device job
to the stream used by DALI.)code", true)
    .NumInput(0, 256)
    .AllowSequences()
    .SupportVolumetric()
    .NoPrune()
    .AddParent("PythonFunctionBase");

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

TensorListShape<> GetDLTensorListShape(const std::vector<DLMTensorPtr>& dl_tensors) {
  TensorListShape<> list_shape{};
  list_shape.resize(dl_tensors.size(), dl_tensors[0]->dl_tensor.ndim);
  for (size_t i = 0; i < dl_tensors.size(); ++i) {
    auto &dl_tensor = dl_tensors[i]->dl_tensor;
    assert(dl_tensor.ndim == list_shape.sample_dim());
    list_shape.set_tensor_shape(i, make_span(dl_tensor.shape, dl_tensor.ndim));
  }
  return list_shape;
}

template <>
void CopyOutputData(TensorVector<CPUBackend> &output, std::vector<DLMTensorPtr> &dl_tensors,
                   int batch_size, HostWorkspace &workspace) {
  auto &thread_pool = workspace.GetThreadPool();
  for (int i = 0; i < batch_size; ++i) {
    thread_pool.DoWorkWithID([&, i](int) {
      CopyDlTensor<CPUBackend>(output[i].raw_mutable_data(), dl_tensors[i]);
    });
  }
  thread_pool.WaitForWork();
}

template <>
void CopyOutputData(TensorList<GPUBackend>& output, std::vector<DLMTensorPtr> &dl_tensors,
                    int batch_size, DeviceWorkspace &workspace) {
  for (int i = 0; i < batch_size; ++i) {
    CopyDlTensor<GPUBackend>(output.raw_mutable_tensor(i), dl_tensors[i], workspace.stream());
  }
}

}  // namespace detail

DALI_REGISTER_OPERATOR(DLTensorPythonFunctionImpl, DLTensorPythonFunctionImpl<CPUBackend>, CPU);

DALI_REGISTER_OPERATOR(DLTensorPythonFunctionImpl, DLTensorPythonFunctionImpl<GPUBackend>, GPU);

}  // namespace dali
