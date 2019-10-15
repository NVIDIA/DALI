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

#include "dali/operators/python_function/python_function.h"
#include <vector>
#include "dali/operators/util/copy_with_stride.h"

namespace dali {

DALI_SCHEMA(PythonFunctionImplBase)
        .AddArg("function_id", R"code(Id of the python function)code", DALI_INT64)
        .AddOptionalArg("num_outputs", R"code(Number of outputs)code", 1)
        .MakeInternal();

DALI_SCHEMA(PythonFunctionImpl)
        .AddParent("PythonFunctionImplBase")
        .DocStr(R"code(This is an auxiliary operator. Use PythonFunction instead.)code")
        .NumInput(0, 256)
        .OutputFn([](const OpSpec &spec) {return spec.GetArgument<int>("num_outputs");})
        .MakeInternal()
        .NoPrune();

DALI_SCHEMA(PythonFunctionBase)
        .AddArg("function",
                R"code(Function object consuming and producing numpy arrays.)code",
                DALI_PYTHON_OBJECT)
        .AddOptionalArg("num_outputs", R"code(Number of outputs)code", 1)
        .MakeInternal();

DALI_SCHEMA(PythonFunction)
        .AddParent("PythonFunctionBase")
        .DocStr("Executes a python function")
        .NumInput(0, 256)
        .NoPrune();

DALI_SCHEMA(TorchPythonFunction)
        .AddParent("PythonFunctionBase")
        .DocStr("Executes a function operating on Torch tensors")
        .NumInput(0, 256)
        .NoPrune();

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

void CopyNumpyArrayToTensor(Tensor<CPUBackend> &tensor, py::array &array) {
  auto buffer_info = array.request();
  TypeInfo type = TypeFromFormatStr(buffer_info.format);
  tensor.set_type(type);
  tensor.Resize(TensorShape<>(array.shape(), array.shape() + array.ndim()));
  CopyWithStride<CPUBackend>(tensor.raw_mutable_data(), buffer_info.ptr,
                             buffer_info.strides.data(), array.shape(),
                             array.ndim(), buffer_info.itemsize);
}

py::list PrepareInputList(SampleWorkspace &ws) {
  py::list args_list;
  for (int i = 0; i < ws.NumInput(); ++i) {
    auto &input = ws.Input<CPUBackend>(i);
    py::dtype dtype(FormatStrFromType(input.type()));
    auto input_array = py::array(dtype, input.shape(), input.raw_data(), py::array());
    args_list.append(input_array);
  }
  return args_list;
}

void CopyOutputs(SampleWorkspace &ws, const py::tuple &output) {
  for (int i = 0; i < ws.NumOutput(); ++i) {
    auto &output_tensor = ws.Output<CPUBackend>(i);
    auto output_array = py::cast<py::array>(output[i]);
    CopyNumpyArrayToTensor(output_tensor, output_array);
  }
}

std::mutex operator_lock{};

template<>
void PythonFunctionImpl<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  std::lock_guard<std::mutex> operator_guard(operator_lock);
  py::gil_scoped_acquire interpreter_guard{};
  py::list args_list = PrepareInputList(ws);
  py::object output_o;
  try {
    output_o = python_function(*py::tuple(args_list));
  } catch(const py::error_already_set &e) {
    throw std::runtime_error(to_string("PythonFunction error: ") + to_string(e.what()));
  }
  if (!output_o.is_none()) {
    py::tuple output = (py::tuple::check_(output_o)) ? output_o : py::make_tuple(output_o);
    DALI_ENFORCE(output.size() == static_cast<size_t>(ws.NumOutput()),
                 "Python function returned " + std::to_string(output.size()) + " outputs and "
                     + std::to_string(ws.NumOutput()) + " were expected.");
    CopyOutputs(ws, output);
  } else {
    DALI_ENFORCE(ws.NumOutput() == 0, "Python function returned 0 outputs and "
        + std::to_string(ws.NumOutput()) + " were expected.");
  }
}

static cudaStream_t current_cuda_stream = nullptr;

cudaStream_t GetCurrentStream() {
  return current_cuda_stream;
}

void SetCurrentStream(cudaStream_t stream) {
  current_cuda_stream = stream;
}

DALI_REGISTER_OPERATOR(PythonFunctionImpl, PythonFunctionImpl<CPUBackend>, CPU);

PYBIND11_MODULE(libpython_function_plugin, m) {
  m.def("current_dali_stream", []() { return reinterpret_cast<uint64_t>(GetCurrentStream()); });
}

}  // namespace dali
