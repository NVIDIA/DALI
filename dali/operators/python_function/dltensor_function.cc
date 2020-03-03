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

#include <pybind11/stl.h>
#include <memory>
#include <utility>
#include "dali/operators/python_function/dltensor_function.h"
#include "dali/pipeline/util/copy_with_stride.h"

namespace dali {

DALI_SCHEMA(DLTensorPythonFunctionImpl)
    .AddOptionalArg("synchronize_stream", "Synchronize CUDA stream", true)
    .AddArg("function_id", R"code(Id of the python function)code", DALI_INT64)
    .AddOptionalArg("num_outputs", R"code(Number of outputs)code", 1)
    .AddArg("batch_processing", "Batch processing.", DALI_BOOL)
    .NumInput(0, 256)
    .OutputFn([](const OpSpec &spec) {return spec.GetArgument<int>("num_outputs");})
    .NoPrune()
    .Unserializable()
    .MakeInternal();

DALI_SCHEMA(DLTensorPythonFunction)
    .DocStr(R"code(Execute a python function that operates on DLPack tensors.
The function should not modify input tensors.

In case of the GPU operator it is a user's responsibility to synchronize the device code with DALI.
This can be accomplished by synchronizing DALI's work before the operator call
with the `synchronize_stream` flag (true by default) and then making sure
the scheduled device tasks are finished within the operator call.
Alternatively, the gpu code can be done on the DALI's stream
which may be determined by calling the `current_dali_stream()` function.
In this case, the `synchronize_stream` flag can be set to false.)code")
    .AddOptionalArg("synchronize_stream",
        R"code(Make DALI synchronize its CUDA stream before calling the python function.
Should be set to false only if the called function schedules the device job
to the stream used by DALI.)code", true)
    .AddOptionalArg("batch_processing",
                    "Whether the function should get the whole batch as input.", true)
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

template <>
py::list PrepareDLTensorInputsPerSample<CPUBackend>(HostWorkspace &ws) {
  py::list input_tuples;
  if (ws.NumInput() == 0) return input_tuples;
  auto batch_size = ws.NumInputAtIdx(0);
  for (Index s = 0; s < batch_size; ++s) {
    py::list tuple;
    for (Index idx = 0; idx < ws.NumInput(); ++idx) {
      auto &t = ws.Input<CPUBackend>(idx, s);
      auto dl_capsule = TensorToDLPackView(const_cast<Tensor<CPUBackend>&>(t));
      tuple.append(dl_capsule);
    }
    input_tuples.append(tuple);
  }
  return input_tuples;
}

template <>
py::list PrepareDLTensorInputsPerSample<GPUBackend>(DeviceWorkspace &ws) {
  std::vector<py::list> input_tuples;
  if (ws.NumInput() == 0) return py::cast(input_tuples);
  Index batch_size = ws.InputRef<GPUBackend>(0).ntensor();
  input_tuples.resize(batch_size);
  for (Index idx = 0; idx < ws.NumInput(); ++idx) {
    py::list dl_tensor_list = TensorListToDLPackView(ws.InputRef<GPUBackend>(idx));
    for (Index s = 0; s < batch_size; ++s) {
      input_tuples[s].append(dl_tensor_list[s]);
    }
  }
  return py::cast(input_tuples);
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

std::mutex operator_lock;

static cudaStream_t current_cuda_stream = nullptr;

cudaStream_t GetCurrentStream() {
  return current_cuda_stream;
}

void SetCurrentStream(cudaStream_t stream) {
  current_cuda_stream = stream;
}

struct PyBindInitializer {
  PyBindInitializer() {
    auto thread_state = PyGILState_Ensure();
    pybind11::get_shared_data("");  // setup the pybind's internals pointer
    PyGILState_Release(thread_state);
  }
};

// Some pybind's internals are not properly initialized when used in dynamically linked library,
// so this workaround initializes them manually
static PyBindInitializer pybind_initializer{}; // NOLINT

struct DLTensorNumpyResource: public DLTensorResource {
  explicit DLTensorNumpyResource(const py::array &array)
      : DLTensorResource(TensorShape<>(array.shape(), array.shape() + array.ndim()))
      , array(array) {
    strides.resize(array.ndim());
    auto itemsize = array.dtype().itemsize();
    for (int i = 0; i < array.ndim(); ++i) {
      strides[i] = array.strides(i) / itemsize;
    }
  }

  py::array array;

  ~DLTensorNumpyResource() override = default;
};

PYBIND11_MODULE(libpython_function_plugin, m) {
  m.def("current_dali_stream", []() { return reinterpret_cast<uint64_t>(GetCurrentStream()); });

  m.def("DLTensorToArray", [](py::capsule dl_capsule) {
    auto dlm_tensor_ptr = DLMTensorPtrFromCapsule(dl_capsule);
    const auto &dl_tensor = dlm_tensor_ptr->dl_tensor;
    auto dali_type = TypeTable::GetTypeInfo(DLToDALIType(dl_tensor.dtype));
    py::dtype dtype(FormatStrFromType(dali_type));
    auto shape = make_span(dl_tensor.shape, dl_tensor.ndim);
    py::array array;
    if (dl_tensor.strides) {
      TensorShape<> strides;
      strides.resize(dl_tensor.ndim);
      for (int i = 0; i < dl_tensor.ndim; ++i) {
        strides[i] = dl_tensor.strides[i] * dtype.itemsize();
      }
      array = py::array(dtype, shape, strides, dl_tensor.data, py::array());
    } else {
      array = py::array(dtype, shape, dl_tensor.data, py::array());
    }
    return array;
  });

  m.def("ArrayToDLTensor", [](py::array array) {
    auto buffer = array.request();
    auto dlm_tensor_ptr = MakeDLTensor(buffer.ptr, TypeFromFormatStr(buffer.format),
                                       false, 0, std::make_unique<DLTensorNumpyResource>(array));
    return DLTensorToCapsule(std::move(dlm_tensor_ptr));
  });
}

}  // namespace dali
