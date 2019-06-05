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

#include "dali/pipeline/operators/python_function/python_function.h"
#include <vector>

namespace dali {

DALI_SCHEMA(PythonFunctionImpl)
        .DocStr(R"code(This is an auxiliary operator. Use PythonFunction instead.)code")
        .NumInput(0, 256)
        .AllowMultipleInputSets()
        .AddArg("function_id", R"code(Id of the python function)code", DALI_INT64)
        .AddOptionalArg("num_outputs", R"code(Number of outputs)code", 1)
        .OutputFn([](const OpSpec &spec) {return spec.GetArgument<int>("num_outputs");})
        .MakeInternal();

DALI_SCHEMA(PythonFunction)
        .DocStr("Executes a python function")
        .NumInput(0, 256)
        .AllowMultipleInputSets()
        .AddArg("function",
                R"code(Function object consuming and producing a single numpy array)code",
                DALI_PYTHON_OBJECT)
        .AddOptionalArg("num_outputs", R"code(Number of outputs)code", 1);

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

py::array NumpyArrayAsContiguous(const TypeInfo &type, const py::array &array) {
  DALI_TYPE_SWITCH(type.id(), DType,
    return py::array_t<DType, py::array::c_style>::ensure(array);
  )
}

void CopyNumpyArrayToTensor(Tensor<CPUBackend> &tensor, py::array &array) {
  std::vector<Index> shape(static_cast<size_t>(array.ndim()));
  std::copy(array.shape(), array.shape() + array.ndim(), shape.begin());
  TypeInfo type = TypeFromFormatStr(array.request().format);
  tensor.set_type(type);
  tensor.Resize(shape);
  py::array contiguous = NumpyArrayAsContiguous(type, array);
  std::memcpy(
      tensor.raw_mutable_data(),
      contiguous.data(),
      static_cast<size_t>(contiguous.size() * contiguous.itemsize()));
}

py::list PrepareInputList(SampleWorkspace *ws, int idx) {
  py::list args_list;
  for (int i = 0; i < ws->NumInput(); ++i) {
    auto &input = ws->Input<CPUBackend>(ws->NumInput() * idx + i);
    py::dtype dtype(FormatStrFromType(input.type()));
    auto input_array = py::array(dtype, input.shape(), input.raw_data(), py::array());
    args_list.append(input_array);
  }
  return args_list;
}

void CopyOutputs(SampleWorkspace *ws, int idx, const py::tuple &output) {
  for (int i = 0; i < ws->NumOutput(); ++i) {
    auto &output_tensor = ws->Output<CPUBackend>(ws->NumInput() * idx + i);
    auto output_array = py::cast<py::array>(output[i]);
    CopyNumpyArrayToTensor(output_tensor, output_array);
  }
}

template<>
void PythonFunctionImpl<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  py::gil_scoped_acquire guard{};
  py::list args_list = PrepareInputList(ws, idx);
  py::object output_o;
  try {
    output_o = python_function(*py::tuple(args_list));
  } catch(const py::error_already_set & e) {
    throw std::runtime_error(to_string("PythonFunction error: ") + to_string(e.what()));
  }
  py::tuple output = (py::tuple::check_(output_o)) ? output_o : py::make_tuple(output_o);
  DALI_ENFORCE(output.size() == static_cast<size_t>(ws->NumOutput()),
               "Python function returned " + std::to_string(output.size()) + " outputs and "
                   + std::to_string(ws->NumOutput()) + " were expected.");
  CopyOutputs(ws, idx, output);
}

DALI_REGISTER_OPERATOR(PythonFunctionImpl, PythonFunctionImpl<CPUBackend>, CPU);

}  // namespace dali
