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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/op_schema.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/python/python3_compat.h"
#include "dali/util/user_stream.h"
#include "dali/pipeline/operators/reader/parser/tfrecord_parser.h"
#include "dali/plugin/copy.h"
#include "dali/plugin/plugin_manager.h"
#include "dali/util/half.hpp"

namespace dali {
namespace python {

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

static std::string FormatStrFromType(TypeInfo type) {
  if (IsType<uint8>(type)) {
    return py::format_descriptor<uint8>::format();
  } else if (IsType<int16>(type)) {
    return py::format_descriptor<int16>::format();
  } else if (IsType<int>(type)) {
    return py::format_descriptor<int>::format();
  } else if (IsType<long>(type)) { // NOLINT
    return py::format_descriptor<long>::format(); // NOLINT
  } else if (IsType<int64>(type)) { // NOLINT
    return py::format_descriptor<long long>::format(); // NOLINT
  } else if (IsType<float>(type)) {
    return py::format_descriptor<float>::format();
  } else if (IsType<double>(type)) {
    return py::format_descriptor<double>::format();
  } else if (IsType<bool>(type)) {
    return py::format_descriptor<bool>::format();
  } else if (IsType<float16>(type)) {
    return "f2";
  } else {
    DALI_FAIL("Cannot convert type " + type.name() +
        " to format descriptor string");
  }
}

static TypeInfo TypeFromFormatStr(std::string format) {
  if (format == py::format_descriptor<uint8>::format()) {
    return TypeInfo::Create<uint8>();
  } else if (format == py::format_descriptor<int16>::format()) {
    return TypeInfo::Create<int16>();
  } else if (format == py::format_descriptor<int>::format()) {
    return TypeInfo::Create<int>();
  } else if (format == py::format_descriptor<long>::format()) { // NOLINT
    return TypeInfo::Create<long>(); // NOLINT
  } else if (format == py::format_descriptor<long long>::format()) { // NOLINT
    return TypeInfo::Create<int64>(); // NOLINT
  } else if (format == py::format_descriptor<float>::format()) {
    return TypeInfo::Create<float>();
  } else if (format == py::format_descriptor<double>::format()) {
    return TypeInfo::Create<double>();
  } else if (format == py::format_descriptor<bool>::format()) {
    return TypeInfo::Create<bool>();
  } else if (format == "f2") {
    return TypeInfo::Create<float16>();
  } else {
    DALI_FAIL("Cannot create type for unknow format string: " + format);
  }
}

void ExposeTensor(py::module &m) { // NOLINT
  py::class_<Tensor<CPUBackend>>(m, "TensorCPU", py::buffer_protocol())
    .def_buffer([](Tensor<CPUBackend> &t) -> py::buffer_info {
          DALI_ENFORCE(IsValidType(t.type()), "Cannot produce "
              "buffer info for tensor w/ invalid type.");

          std::vector<ssize_t> shape(t.ndim()), stride(t.ndim());
          size_t dim_prod = 1;
          for (int i = 0; i < t.ndim(); ++i) {
            shape[i] = t.shape()[i];

            // We iterate over stride backwards
            stride[(t.ndim()-1) - i] = t.type().size()*dim_prod;
            dim_prod *= t.shape()[(t.ndim()-1) - i];
          }

          return py::buffer_info(
              t.raw_mutable_data(),
              t.type().size(),
              FormatStrFromType(t.type()),
              t.ndim(), shape, stride);
        })
    .def("__init__", [](Tensor<CPUBackend> &t, py::buffer b) {
          // We need to verify that hte input data is c contiguous
          // and of a type that we can work with in the backend
          py::buffer_info info = b.request();

          std::vector<Index> i_shape;
          for (auto &dim : info.shape) {
            i_shape.push_back(dim);
          }
          size_t bytes = volume(i_shape) * info.itemsize;

          // Validate the stride
          ssize_t dim_prod = 1;
          for (int i = info.strides.size()-1; i >= 0; --i) {
            DALI_ENFORCE(info.strides[i] == info.itemsize*dim_prod,
                "Strided data not supported. Detected on dimension " + std::to_string(i));
            dim_prod *= info.shape[i];
          }

          // Create the Tensor and wrap the data
          new (&t) Tensor<CPUBackend>;
          TypeInfo type = TypeFromFormatStr(info.format);
          t.ShareData(info.ptr, bytes);
          t.set_type(type);
          t.Resize(i_shape);
        },
      R"code(
      Tensor residing in the CPU memory.
      )code")
    .def("shape", &Tensor<CPUBackend>::shape,
         R"code(
         Shape of the tensor.
         )code")
    .def("squeeze", &Tensor<CPUBackend>::Squeeze,
         R"code(
         Remove single-dimensional entries from the shape of the Tensor.
         )code")
    .def("copy_to_external",
        [](Tensor<CPUBackend> &t, py::object p) {
          PyObject *p_ptr = p.ptr();
          PyObject *ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
          void *ptr = PyLong_AsVoidPtr(ptr_as_int);
          CopyToExternalTensor(t, ptr, CPU);
        },
      "ptr"_a,
      R"code(
      Copy to external pointer in the CPU memory.

      Parameters
      ----------
      ptr : ctypes.c_void_p
            Destination of the copy.
      )code")
    .def("dtype",
        [](Tensor<CPUBackend> &t) {
          return FormatStrFromType(t.type());
        },
      R"code(
      String representing NumPy type of the Tensor.
      )code");

  py::class_<Tensor<GPUBackend>>(m, "TensorGPU")
    .def("shape", &Tensor<GPUBackend>::shape,
         R"code(
         Shape of the tensor.
         )code")
    .def("squeeze", &Tensor<GPUBackend>::Squeeze,
         R"code(
         Remove single-dimensional entries from the shape of the Tensor.
         )code")
    .def("copy_to_external",
        [](Tensor<GPUBackend> &t, py::object p) {
          PyObject *p_ptr = p.ptr();
          PyObject *ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
          void *ptr = PyLong_AsVoidPtr(ptr_as_int);
          CopyToExternalTensor(t, ptr, GPU);
        },
      "ptr"_a,
      R"code(
      Copy to external pointer in the GPU memory.

      Parameters
      ----------
      ptr : ctypes.c_void_p
            Destination of the copy.
      )code")
    .def("dtype",
        [](Tensor<GPUBackend> &t) {
          return FormatStrFromType(t.type());
        },
      R"code(
      String representing NumPy type of the Tensor.
      )code");
}

void ExposeTensorList(py::module &m) { // NOLINT
  // We only want to wrap buffers w/ TensorLists to feed then to
  // the backend. We do not support converting from TensorLists
  // to numpy arrays currently.
  py::class_<TensorList<CPUBackend>>(m, "TensorListCPU", py::buffer_protocol())
    .def("__init__", [](TensorList<CPUBackend> &t, py::buffer b) {
          // We need to verify that the input data is C_CONTIGUOUS
          // and of a type that we can work with in the backend
          py::buffer_info info = b.request();

          DALI_ENFORCE(info.shape.size() > 0,
              "Cannot create TensorList from 0-dim array.");

          // Create a list of shapes
          std::vector<Index> tensor_shape(info.shape.size()-1);
          for (size_t i = 1; i < info.shape.size(); ++i) {
            tensor_shape[i-1] = info.shape[i];
          }
          std::vector<Dims> i_shape(info.shape[0], tensor_shape);
          size_t bytes = volume(tensor_shape)*i_shape.size()*info.itemsize;

          // Validate the stride
          ssize_t dim_prod = 1;
          for (int i = info.strides.size()-1; i >= 0; --i) {
            DALI_ENFORCE(info.strides[i] == info.itemsize*dim_prod,
                "Strided data not supported. Detected on dimension " + std::to_string(i));
            dim_prod *= info.shape[i];
          }

          // Create the Tensor and wrap the data
          new (&t) TensorList<CPUBackend>;
          TypeInfo type = TypeFromFormatStr(info.format);
          t.ShareData(info.ptr, bytes);
          t.set_type(type);
          t.Resize(i_shape);
        },
      R"code(
      List of tensors residing in the CPU memory.

      Parameters
      ----------
      )code")
    .def("at", [](TensorList<CPUBackend> &t, Index id) -> py::array {
          DALI_ENFORCE(IsValidType(t.type()), "Cannot produce "
              "buffer info for tensor w/ invalid type.");
          DALI_ENFORCE(static_cast<size_t>(id) < t.ntensor(), "Index is out-of-range.");
          DALI_ENFORCE(id >= 0, "Index is out-of-range.");

          std::vector<ssize_t> shape(t.tensor_shape(id).size()), stride(t.tensor_shape(id).size());
          size_t dim_prod = 1;
          for (size_t i = 0; i < shape.size(); ++i) {
            shape[i] = t.tensor_shape(id)[i];

            // We iterate over stride backwards
            stride[(stride.size()-1) - i] = t.type().size()*dim_prod;
            dim_prod *= t.tensor_shape(id)[(shape.size()-1) - i];
          }

          return py::array(py::buffer_info(
              t.raw_mutable_tensor(id),
              t.type().size(),
              FormatStrFromType(t.type()),
              shape.size(), shape, stride));
        },
      R"code(
      Returns tensor at given position in the list.

      Parameters
      ----------
      )code")
    .def("as_array", [](TensorList<CPUBackend> &t) -> py::array {
          void* raw_mutable_data = nullptr;
          std::string format;
          size_t type_size;

          if (t.size() > 0) {
            DALI_ENFORCE(IsValidType(t.type()), "Cannot produce "
                "buffer info for tensor w/ invalid type.");
            DALI_ENFORCE(t.IsDenseTensor(),
                        "Tensors in the list must have the same shape");
            raw_mutable_data = t.raw_mutable_data();
          }

          if (IsValidType(t.type())) {
            format = FormatStrFromType(t.type());
            type_size = t.type().size();
          } else {
            // Default is float
            format = py::format_descriptor<float>::format();
            type_size = sizeof(float);
          }

          auto shape_size = t.shape().size() > 0 ? t.tensor_shape(0).size() : 0;
          std::vector<ssize_t> shape(shape_size + 1);
          std::vector<ssize_t> strides(shape_size + 1);
          size_t dim_prod = 1;
          for (size_t i = 0; i < shape.size(); ++i) {
            if (i == 0) {
              shape[i] = t.shape().size();
            } else {
              shape[i] = t.tensor_shape(0)[i - 1];
            }

            // We iterate over stride backwards
            strides[(strides.size()-1) - i] = type_size*dim_prod;
            if (i == shape.size() - 1) {
              dim_prod *= t.shape().size();
            } else {
              dim_prod *= t.tensor_shape(0)[(shape.size()-2) - i];
            }
          }

          return py::array(py::buffer_info(
              raw_mutable_data,
              type_size,
              std::move(format),
              shape.size(), shape, strides));
        },
      R"code(
      Returns TensorList as a numpy array. TensorList must be dense.

      Parameters
      ----------
      )code")
    .def("__len__", [](TensorList<CPUBackend> &t) {
          return t.ntensor();
        })
    .def("is_dense_tensor", &TensorList<CPUBackend>::IsDenseTensor,
      R"code(
      Checks whether all tensors in this `TensorList` have the same shape
      (and so the list itself can be viewed as a tensor).

      For example, if `TensorList` contains `N` tensors, each with shape
      `(H,W,C)` (with the same values of `H`, `W` and `C`), then the list
      may be viewed as a tensor of shape `(N, H, W, C)`.
      )code")
    .def("copy_to_external",
        [](TensorList<CPUBackend> &t, py::object p) {
          PyObject *p_ptr = p.ptr();
          PyObject *ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
          void *ptr = PyLong_AsVoidPtr(ptr_as_int);
          CopyToExternalTensor(&t, ptr, CPU);
        },
      R"code(
      Copy the contents of this `TensorList` to an external pointer
      (of type `ctypes.c_void_p`) residing in CPU memory.

      This function is used internally by plugins to interface with
      tensors from supported Deep Learning frameworks.

      Parameters
      ----------
      )code")
    .def("as_tensor", &TensorList<CPUBackend>::AsTensor,
      R"code(
      Returns a tensor that is a view of this `TensorList`.

      This function can only be called if `is_dense_tensor` returns `True`.
      )code",
      py::return_value_policy::reference_internal);

  py::class_<TensorList<GPUBackend>>(m, "TensorListGPU", py::buffer_protocol())
    .def("__init__", [](TensorList<GPUBackend> &t) {
          // Construct a default TensorList on GPU
          new (&t) TensorList<GPUBackend>;
        },
      R"code(
      List of tensors residing in the GPU memory.
      )code")
    .def("as_cpu", [](TensorList<GPUBackend> &t) -> TensorList<CPUBackend>* {
          TensorList<CPUBackend> * ret = new TensorList<CPUBackend>();
          ret->set_pinned(false);
          UserStream * us = UserStream::Get();
          cudaStream_t s = us->GetStream(t);
          ret->Copy(t, s);
          CUDA_CALL(cudaStreamSynchronize(s));
          return ret;
        },
      R"code(
      Returns a `TensorListCPU` object being a copy of this `TensorListGPU`.
      )code",
      py::return_value_policy::take_ownership)
    .def("__len__", [](TensorList<GPUBackend> &t) {
          return t.ntensor();
        })
    .def("is_dense_tensor", &TensorList<GPUBackend>::IsDenseTensor,
      R"code(
      Checks whether all tensors in this `TensorList` have the same shape
      (and so the list itself can be viewed as a tensor).

      For example, if `TensorList` contains `N` tensors, each with shape
      `(H,W,C)` (with the same values of `H`, `W` and `C`), then the list
      may be viewed as a tensor of shape `(N, H, W, C)`.
      )code")
    .def("copy_to_external",
        [](TensorList<GPUBackend> &t, py::object p) {
          PyObject *p_ptr = p.ptr();
          PyObject *ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
          void *ptr = PyLong_AsVoidPtr(ptr_as_int);
          CopyToExternalTensor(&t, ptr, GPU);
        },
      R"code(
      Copy the contents of this `TensorList` to an external pointer
      (of type `ctypes.c_void_p`) residing in CPU memory.

      This function is used internally by plugins to interface with
      tensors from supported Deep Learning frameworks.

      Parameters
      ----------
      )code")
    .def("at",
        [](TensorList<GPUBackend> &t, Index id) -> std::unique_ptr<Tensor<GPUBackend>> {
          std::unique_ptr<Tensor<GPUBackend>> ptr(new Tensor<GPUBackend>());
          ptr->ShareData(&t, id);
          return ptr;
        },
      R"code(
      Returns a tensor at given position in the list.

      Parameters
      ----------
      )code",
      py::keep_alive<0, 1>())
    .def("as_tensor", &TensorList<GPUBackend>::AsTensor,
      R"code(
      Returns a tensor that is a view of this `TensorList`.

      This function can only be called if `is_dense_tensor` returns `True`.
      )code",
      py::return_value_policy::reference_internal);
}

#define GetRegisteredOpsFor(OPTYPE)                                           \
static vector<string> GetRegistered##OPTYPE##Ops(bool internal_ops = false) { \
  return OPTYPE##OperatorRegistry::Registry().RegisteredNames(internal_ops);  \
}
GetRegisteredOpsFor(CPU)
GetRegisteredOpsFor(GPU)
GetRegisteredOpsFor(Mixed)
GetRegisteredOpsFor(Support)
#undef GetRegisteredOpsFor

static const OpSchema &GetSchema(const string &name) {
  return SchemaRegistry::GetSchema(name);
}

static constexpr int GetCxx11AbiFlag() {
#ifdef _GLIBCXX_USE_CXX11_ABI
  return _GLIBCXX_USE_CXX11_ABI;
#else
  return 0;
#endif
}

#ifdef DALI_BUILD_PROTO3
typedef dali::TFRecordParser::FeatureType TFFeatureType;
typedef dali::TFRecordParser::Feature TFFeature;
typedef TFFeature::Value TFValue;

TFValue ConvertTFRecordDefaultValue(TFFeatureType type, py::object val) {
  PyObject *ptr = val.ptr();
  TFValue ret;
  switch (type) {
    case TFFeatureType::int64:
      DALI_ENFORCE(PyInt_Check(ptr) || PyLong_Check(ptr),
          "Invalid type for default value, expected int.");
      ret.int64 = PyInt_AsLong(ptr);
      break;
    case TFFeatureType::string:
      DALI_ENFORCE(PyStr_Check(ptr),
          "Invalid type for default value, expected string.");
      ret.str = PyStr_AsString(ptr);
      break;
    case TFFeatureType::float32:
      DALI_ENFORCE(PyFloat_Check(ptr),
          "Invalid type for default value, expected float.");
      ret.float32 = PyFloat_AsDouble(ptr);
      break;
    default:
      DALI_FAIL("Invalid type for default value, expected string, int or float.");
  }
  return ret;
}
#endif  // DALI_BUILD_PROTO3

PYBIND11_MODULE(backend_impl, m) {
  m.doc() = "Python bindings for the C++ portions of DALI";

  // DALI Init function
  m.def("Init", &DALIInit);

  m.def("LoadLibrary", &PluginManager::LoadLibrary);

  m.def("GetCxx11AbiFlag", &GetCxx11AbiFlag);

  // Types
  py::module types_m = m.def_submodule("types");
  types_m.doc() = "Datatypes and options used by DALI";

  // DALIDataType
  py::enum_<DALIDataType>(types_m, "DALIDataType", "Data type of image")
    .value("NO_TYPE", DALI_NO_TYPE)
    .value("UINT8", DALI_UINT8)
    .value("FLOAT16", DALI_FLOAT16)
    .value("FLOAT", DALI_FLOAT)
    .value("INT64", DALI_INT64)
    .value("INT32", DALI_INT32)
    .value("INT16", DALI_INT16)
    .value("BOOL", DALI_BOOL)
    .value("STRING", DALI_STRING)
    .value("_BOOL_VEC", DALI_BOOL_VEC)
    .value("_INT32_VEC", DALI_INT_VEC)
    .value("_STRING_VEC", DALI_STRING_VEC)
    .value("_FLOAT_VEC", DALI_FLOAT_VEC)
#ifdef DALI_BUILD_PROTO3
    .value("FEATURE", DALI_TF_FEATURE)
    .value("_FEATURE_VEC", DALI_TF_FEATURE_VEC)
    .value("_FEATURE_DICT", DALI_TF_FEATURE_DICT)
#endif  // DALI_BUILD_PROTO3
    .value("IMAGE_TYPE", DALI_IMAGE_TYPE)
    .value("DATA_TYPE", DALI_DATA_TYPE)
    .value("INTERP_TYPE", DALI_INTERP_TYPE)
    .value("TENSOR_LAYOUT", DALI_TENSOR_LAYOUT)
    .export_values();

  // DALIImageType
  py::enum_<DALIImageType>(types_m, "DALIImageType", "Image type")
    .value("RGB", DALI_RGB)
    .value("BGR", DALI_BGR)
    .value("GRAY", DALI_GRAY)
    .value("YCbCr", DALI_YCbCr)
    .export_values();

  // DALIInterpType
  py::enum_<DALIInterpType>(types_m, "DALIInterpType", "Interpolation mode")
    .value("INTERP_NN", DALI_INTERP_NN)
    .value("INTERP_LINEAR", DALI_INTERP_LINEAR)
    .value("INTERP_CUBIC", DALI_INTERP_CUBIC)
    .value("INTERP_LANCZOS3", DALI_INTERP_LANCZOS3)
    .value("INTERP_TRIANGULAR", DALI_INTERP_TRIANGULAR)
    .value("INTERP_GAUSSIAN", DALI_INTERP_GAUSSIAN)
    .export_values();

  // DALITensorLayout
  py::enum_<DALITensorLayout>(types_m, "DALITensorLayout", "Tensor layout")
    .value("NCHW", DALI_NCHW)
    .value("NHWC", DALI_NHWC)
    .value("NFHWC", DALI_NFHWC)
    .value("NFCHW", DALI_NFCHW)
    .value("SAME", DALI_SAME)
    .export_values();

  // Operator node
  py::class_<OpNode>(m, "OpNode")
    .def("instance_name",
        [](OpNode* node) {
          return node->instance_name;
        })
    .def("name",
        [](OpNode* node) {
          return node->spec.name();
        });

  // Pipeline class
  py::class_<Pipeline>(m, "Pipeline")
    .def(py::init(
            [](int batch_size, int num_threads, int device_id, int64_t seed = -1,
                bool pipelined_execution = true, int prefetch_queue_depth = 2,
                bool async_execution = true, size_t bytes_per_sample_hint = 0,
                bool set_affinity = false, int max_num_stream = -1,
                int default_cuda_stream_priority = 0) {
              return std::unique_ptr<Pipeline>(
                  new Pipeline(batch_size, num_threads, device_id, seed, pipelined_execution,
                      prefetch_queue_depth, async_execution, bytes_per_sample_hint, set_affinity,
                      max_num_stream, default_cuda_stream_priority));
            }),
        "batch_size"_a,
        "num_threads"_a,
        "device_id"_a,
        "seed"_a,
        "exec_pipelined"_a,
        "prefetch_queue_depth"_a = 2,
        "exec_async"_a,
        "bytes_per_sample_hint"_a = 0,
        "set_affinity"_a = false,
        "max_num_stream"_a = -1,
        "default_cuda_stream_priority"_a = 0
        )
    // initialize from serialized pipeline
    .def(py::init(
          [](string serialized_pipe,
             int batch_size, int num_threads, int device_id,
             bool pipelined_execution = true,  int prefetch_queue_depth = 2,
             bool async_execution = true, size_t bytes_per_sample_hint = 0,
             bool set_affinity = false, int max_num_stream = -1,
             int default_cuda_stream_priority = 0) {
              return std::unique_ptr<Pipeline>(
                  new Pipeline(serialized_pipe,
                               batch_size, num_threads, device_id, pipelined_execution,
                               prefetch_queue_depth, async_execution, bytes_per_sample_hint,
                               set_affinity, max_num_stream, default_cuda_stream_priority));
            }),
        "serialized_pipe"_a,
        "batch_size"_a,
        "num_threads"_a,
        "device_id"_a,
        "exec_pipelined"_a,
        "prefetch_queue_depth"_a = 2,
        "exec_async"_a,
        "bytes_per_sample_hint"_a = 0,
        "set_affinity"_a = false,
        "max_num_stream"_a = -1,
        "default_cuda_stream_priority"_a = 0
        )
    .def("AddOperator", &Pipeline::AddOperator)
    .def("GetOperatorNode", &Pipeline::GetOperatorNode)
    .def("Build",
        [](Pipeline *p, const std::vector<std::pair<string, string>>& outputs) {
          p->Build(outputs);
          })
    .def("Build",
        [](Pipeline *p) {
          p->Build();
          })
    .def("SetExecutionTypes",
        [](Pipeline *p, bool exec_pipelined, bool exec_separated, bool exec_async) {
          p->SetExecutionTypes(exec_pipelined, exec_separated, exec_async);
        },
        "exec_pipelined"_a = true,
        "exec_separated"_a = false,
        "exec_async"_a = true)
    .def("SetQueueSizes",
        [](Pipeline *p, int cpu_size, int gpu_size) {
          p->SetQueueSizes(cpu_size, gpu_size);
        })
    .def("SetOutputNames",
        [](Pipeline *p, const std::vector<std::pair<string, string>>& outputs) {
          p->SetOutputNames(outputs);
          })
    .def("RunCPU", &Pipeline::RunCPU)
    .def("RunGPU", &Pipeline::RunGPU)
    .def("Outputs",
        [](Pipeline *p) {
          DeviceWorkspace ws;
          p->Outputs(&ws);

          py::list list;
          for (int i = 0; i < ws.NumOutput(); ++i) {
            if (ws.OutputIsType<CPUBackend>(i)) {
              list.append(&ws.Output<CPUBackend>(i));
            } else {
              list.append(&ws.Output<GPUBackend>(i));
            }
          }
          return list;
        }, py::return_value_policy::take_ownership)
    .def("ShareOutputs",
        [](Pipeline *p) {
          DeviceWorkspace ws;
          p->ShareOutputs(&ws);

          py::list list;
          for (int i = 0; i < ws.NumOutput(); ++i) {
            if (ws.OutputIsType<CPUBackend>(i)) {
              list.append(&ws.Output<CPUBackend>(i));
            } else {
              list.append(&ws.Output<GPUBackend>(i));
            }
          }
          return list;
        }, py::return_value_policy::take_ownership)
    .def("ReleaseOutputs",
        [](Pipeline *p) {
          p->ReleaseOutputs();
        })
    .def("batch_size", &Pipeline::batch_size)
    .def("num_threads", &Pipeline::num_threads)
    .def("device_id", &Pipeline::device_id)
    .def("SetExternalTLInput",
        [](Pipeline *p, const string &name, const TensorList<CPUBackend> &tl) {
          p->SetExternalInput(name, tl);
        })
    .def("SetExternalTensorInput",
        [](Pipeline *p, const string &name, py::list list) {
          // Note: This is a hack to get around weird casting
          // issues w/ pybind and a non-copyable type (dali::Tensor).
          // We cannot use pybind::cast<Tensor<CPUBackend>>
          // because somewhere through the chain of templates
          // pybind returns the calling template type, which
          // tries to call the deleted copy constructor for Tensor.
          // instead, we cast to a reference type and manually
          // move into the vector.
          vector<Tensor<CPUBackend>> tensors(list.size());
          for (size_t i = 0; i < list.size(); ++i) {
            tensors[i] = std::move(list[i].cast<Tensor<CPUBackend>&>());
          }
          p->SetExternalInput(name, tensors);
        })
    .def("SerializeToProtobuf",
        [](Pipeline *p) -> py::bytes {
          string s = p->SerializeToProtobuf();
          return s;
          }, py::return_value_policy::take_ownership)
    .def("SaveGraphToDotFile",
        [](Pipeline *p, const string &filename) {
          p->SaveGraphToDotFile(filename);
        })
    .def("epoch_size", &Pipeline::EpochSize)
    .def("epoch_size",
        [](Pipeline* p, const std::string& op_name) {
          std::map<std::string, Index> sizes = p->EpochSize();
          DALI_ENFORCE(sizes.find(op_name) != sizes.end(),
              "Operator " + op_name + " does not expose valid epoch size.");
          return sizes[op_name];
        });

#define DALI_OPSPEC_ADDARG(T) \
    .def("AddArg", \
        [](OpSpec *spec, const string& name, T v) -> OpSpec& { \
        spec->AddArg(name, v); \
        return *spec; \
      }, py::return_value_policy::reference_internal) \
    .def("AddArg", \
        [](OpSpec *spec, const string& name, std::vector<T> v) -> OpSpec& { \
        spec->AddArg(name, v); \
        return *spec; \
      }, py::return_value_policy::reference_internal)

  py::class_<OpSpec>(m, "OpSpec")
    .def(py::init<std::string>(), "name"_a)
    .def("AddInput", &OpSpec::AddInput,
        "name"_a,
        "device"_a,
        "regular_input"_a = true,
        py::return_value_policy::reference_internal)
    .def("AddArgumentInput", &OpSpec::AddArgumentInput,
        py::return_value_policy::reference_internal)
    .def("AddOutput", &OpSpec::AddOutput,
        py::return_value_policy::reference_internal)
    DALI_OPSPEC_ADDARG(std::string)
    DALI_OPSPEC_ADDARG(bool)
    DALI_OPSPEC_ADDARG(int64)
    DALI_OPSPEC_ADDARG(float)
#ifdef DALI_BUILD_PROTO3
    DALI_OPSPEC_ADDARG(TFFeature)
#endif
    .def("AddArg",
        [](OpSpec *spec, const string &name, py::object obj) -> OpSpec& {
          DALI_FAIL("Unsupported argument type with name " + name);
          return *spec;
        }, py::return_value_policy::reference_internal)
    .def("__repr__", &OpSpec::ToString)
    .def("copy", [](OpSpec &o) -> OpSpec * {
        OpSpec * ret = new OpSpec(o);
        return ret;
        }, py::return_value_policy::take_ownership);

  // Registries for cpu, gpu & mixed operators
  m.def("RegisteredCPUOps", &GetRegisteredCPUOps, py::arg("internal_ops") = false);
  m.def("RegisteredGPUOps", &GetRegisteredGPUOps, py::arg("internal_ops") = false);
  m.def("RegisteredMixedOps", &GetRegisteredMixedOps, py::arg("internal_ops") = false);
  m.def("RegisteredSupportOps", &GetRegisteredSupportOps, py::arg("internal_ops") = false);

  // Registry for OpSchema
  m.def("GetSchema", &GetSchema, py::return_value_policy::reference);

  py::class_<OpSchema>(m, "OpSchema")
    .def("Dox", &OpSchema::Dox)
    .def("MaxNumInput", &OpSchema::MaxNumInput)
    .def("MinNumInput", &OpSchema::MinNumInput)
    .def("HasOutputFn", &OpSchema::HasOutputFn)
    .def("CalculateOutputs", &OpSchema::CalculateOutputs)
    .def("CalculateAdditionalOutputs", &OpSchema::CalculateAdditionalOutputs)
    .def("SupportsInPlace", &OpSchema::SupportsInPlace)
    .def("CheckArgs", &OpSchema::CheckArgs)
    .def("GetArgumentDox", &OpSchema::GetArgumentDox)
    .def("GetArgumentType", &OpSchema::GetArgumentType)
    .def("GetArgumentDefaultValueString", &OpSchema::GetArgumentDefaultValueString)
    .def("GetArgumentNames", &OpSchema::GetArgumentNames)
    .def("IsArgumentOptional", &OpSchema::HasOptionalArgument,
        "arg_name"_a,
        "local_only"_a = false)
    .def("IsTensorArgument", &OpSchema::IsTensorArgument)
    .def("IsSequenceOperator", &OpSchema::IsSequenceOperator)
    .def("AllowsSequences", &OpSchema::AllowsSequences)
    .def("IsInternal", &OpSchema::IsInternal);

  ExposeTensor(m);
  ExposeTensorList(m);

#ifdef DALI_BUILD_PROTO3
  // TFRecord
  py::module tfrecord_m = m.def_submodule("tfrecord");
  tfrecord_m.doc() = "Additional data structures and constants for TFRecord file format support";
  tfrecord_m.attr("int64") = static_cast<int>(TFFeatureType::int64);
  tfrecord_m.attr("string") = static_cast<int>(TFFeatureType::string);
  tfrecord_m.attr("float32") = static_cast<int>(TFFeatureType::float32);

  py::class_<TFFeature>(tfrecord_m, "Feature");

  tfrecord_m.def("FixedLenFeature",
      [](vector<Index> converted_shape, int type, py::object default_value) {
        TFFeatureType converted_type = static_cast<TFFeatureType>(type);
        TFValue converted_default_value =
          ConvertTFRecordDefaultValue(converted_type, default_value);
        return new TFFeature(converted_shape, converted_type, converted_default_value);
      });
  tfrecord_m.def("VarLenFeature",
      [](int type, py::object default_value) {
        TFFeatureType converted_type = static_cast<TFFeatureType>(type);
        TFValue converted_default_value =
          ConvertTFRecordDefaultValue(converted_type, default_value);
        return new TFFeature(converted_type, converted_default_value);
      });
#endif  // DALI_BUILD_PROTO3
}

}  // namespace python
}  // namespace dali
