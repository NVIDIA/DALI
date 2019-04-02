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

DALI_SCHEMA(PythonFunction)
        .DocStr(R"code(Executes python function that consumes and produces single numpy array.)code")
        .NumInput(1)
        .NumOutput(1)
        .AllowMultipleInputSets()
        .EnforceInputLayout(DALI_NHWC)
        .AddArg("function_id",
                R"code(Id of the python function.)code",
                DALI_INT64);

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

py::array tensorToNumpyArray(const Tensor<CPUBackend> &tensor) {
  DALI_TYPE_SWITCH(tensor.type().id(), DType,
      return py::array_t<DType, py::array::c_style>(tensor.shape(), tensor.template data<DType>());
  )
}

TypeInfo getDaliType(const py::array &array) {
  if (array.dtype().kind() == 'i') {
    if (array.itemsize() == sizeof(int16)) {
      return TypeInfo::Create<int16>();
    } else if (array.itemsize() == sizeof(int32)) {
      return TypeInfo::Create<int32>();
    } else if (array.itemsize() == sizeof(int64)) {
      return TypeInfo::Create<int64>();
    }
  } else if (array.dtype().kind() == 'u') {
    if (array.itemsize() == sizeof(uint8)) {
      return TypeInfo::Create<uint8>();
    }
  } else if (array.dtype().kind() == 'f') {
    if (array.itemsize() == sizeof(float16)) {
      return TypeInfo::Create<float16>();
    } else if (array.itemsize() == sizeof(float)) {
      return TypeInfo::Create<float>();
    } else if (array.itemsize() == sizeof(double)) {
      return TypeInfo::Create<double>();
    }
  } else if (array.dtype().kind() == 'b') {
    return TypeInfo::Create<bool>();
  }
  DALI_FAIL("Unexpected numpy array type.");
}

py::array numpyArrayAsContiguous(const TypeInfo &type, const py::array &array) {
  DALI_TYPE_SWITCH(type.id(), DType,
                   return py::array_t<DType, py::array::c_style>::ensure(array);
  )
}

void copyNumpyArrayToTensor(const py::array &array, Tensor<CPUBackend> &tensor) {
  std::vector<Index> shape(static_cast<size_t>(array.ndim()));
  std::copy(array.shape(), array.shape() + array.ndim(), shape.begin());
  TypeInfo type = getDaliType(array);
  tensor.set_type(type);
  tensor.Resize(shape);
  py::array contiguous = numpyArrayAsContiguous(type, array);
  std::memcpy(
      tensor.raw_mutable_data(),
      contiguous.data(),
      static_cast<size_t>(contiguous.size() * contiguous.itemsize()));
}

template<>
void PythonFunction<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(idx);
  py::gil_scoped_acquire guard{};
  py::array output_array = python_function(tensorToNumpyArray(input));
  copyNumpyArrayToTensor(output_array, output);
}

DALI_REGISTER_OPERATOR(PythonFunction, PythonFunction<CPUBackend>, CPU);

}  // namespace dali
