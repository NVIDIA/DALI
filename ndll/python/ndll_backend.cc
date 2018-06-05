// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ndll/pipeline/init.h"
#include "ndll/pipeline/operators/operator.h"
#include "ndll/pipeline/operators/op_schema.h"
#include "ndll/pipeline/operators/op_spec.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"
#include "ndll/python/python3_compat.h"
#include "ndll/util/user_stream.h"
#include "ndll/pipeline/operators/reader/parser/tfrecord_parser.h"
#include "ndll/plugin/copy.h"

namespace ndll {
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
  } else {
    NDLL_FAIL("Cannot convert type " + type.name() +
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
  } else {
    NDLL_FAIL("Cannot create type for unknow format string: " + format);
  }
}

void ExposeTensor(py::module &m) { // NOLINT
  py::class_<Tensor<CPUBackend>>(m, "TensorCPU", py::buffer_protocol())
    .def_buffer([](Tensor<CPUBackend> &t) -> py::buffer_info {
          NDLL_ENFORCE(IsValidType(t.type()), "Cannot produce "
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
          size_t bytes = Product(i_shape) * info.itemsize;

          // Validate the stride
          ssize_t dim_prod = 1;
          for (int i = info.strides.size()-1; i >= 0; --i) {
            NDLL_ENFORCE(info.strides[i] == info.itemsize*dim_prod,
                "Strided data not supported. Detected on dimension " + std::to_string(i));
            dim_prod *= info.shape[i];
          }

          // Create the Tensor and wrap the data
          new (&t) Tensor<CPUBackend>;
          TypeInfo type = TypeFromFormatStr(info.format);
          t.ShareData(info.ptr, bytes);
          t.set_type(type);
          t.Resize(i_shape);
        })
    .def("shape", &Tensor<CPUBackend>::shape)
    .def("ndim", &Tensor<CPUBackend>::ndim)
    .def("dim", &Tensor<CPUBackend>::dim)
    .def("resize", &Tensor<CPUBackend>::Resize)
    .def("squeeze", &Tensor<CPUBackend>::Squeeze)
    .def("copy_to_external",
        [](Tensor<CPUBackend> &t, py::object p) {
          PyObject *p_ptr = p.ptr();
          PyObject *ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
          void *ptr = PyLong_AsVoidPtr(ptr_as_int);
          CopyToExternalTensor(t, ptr);
        })
    .def("dtype",
        [](Tensor<CPUBackend> &t) {
          return FormatStrFromType(t.type());
        });

  py::class_<Tensor<GPUBackend>>(m, "TensorGPU")
    .def("shape", &Tensor<GPUBackend>::shape)
    .def("ndim", &Tensor<GPUBackend>::ndim)
    .def("dim", &Tensor<GPUBackend>::dim)
    .def("resize", &Tensor<GPUBackend>::Resize)
    .def("squeeze", &Tensor<GPUBackend>::Squeeze)
    .def("copy_to_external",
        [](Tensor<GPUBackend> &t, py::object p) {
          PyObject *p_ptr = p.ptr();
          PyObject *ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
          void *ptr = PyLong_AsVoidPtr(ptr_as_int);
          CopyToExternalTensor(t, ptr);
        })
    .def("dtype",
        [](Tensor<GPUBackend> &t) {
          return FormatStrFromType(t.type());
        });
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

          NDLL_ENFORCE(info.shape.size() > 0,
              "Cannot create TensorList from 0-dim array.");

          // Create a list of shapes
          std::vector<Index> tensor_shape(info.shape.size()-1);
          for (size_t i = 1; i < info.shape.size(); ++i) {
            tensor_shape[i-1] = info.shape[i];
          }
          std::vector<Dims> i_shape(info.shape[0], tensor_shape);
          size_t bytes = Product(tensor_shape)*i_shape.size()*info.itemsize;

          // Validate the stride
          ssize_t dim_prod = 1;
          for (int i = info.strides.size()-1; i >= 0; --i) {
            NDLL_ENFORCE(info.strides[i] == info.itemsize*dim_prod,
                "Strided data not supported. Detected on dimension " + std::to_string(i));
            dim_prod *= info.shape[i];
          }

          // Create the Tensor and wrap the data
          new (&t) TensorList<CPUBackend>;
          TypeInfo type = TypeFromFormatStr(info.format);
          t.ShareData(info.ptr, bytes);
          t.set_type(type);
          t.Resize(i_shape);
        })
    .def("at", [](TensorList<CPUBackend> &t, Index id) -> py::array {
          NDLL_ENFORCE(IsValidType(t.type()), "Cannot produce "
              "buffer info for tensor w/ invalid type.");
          NDLL_ENFORCE(id < t.ntensor(), "Index is out-of-range.");
          NDLL_ENFORCE(id >= 0, "Index is out-of-range.");

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
        })
    .def("__len__", [](TensorList<CPUBackend> &t) {
          return t.ntensor();
        })
    .def("is_dense_tensor", &TensorList<CPUBackend>::IsDenseTensor)
    .def("copy_to_external",
        [](TensorList<CPUBackend> &t, py::object p) {
          PyObject *p_ptr = p.ptr();
          PyObject *ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
          void *ptr = PyLong_AsVoidPtr(ptr_as_int);
          CopyToExternalTensor(&t, ptr);
        })
    .def("as_tensor",
        [](TensorList<CPUBackend> &t) -> Tensor<CPUBackend>* {
          Tensor<CPUBackend> * ret = new Tensor<CPUBackend>();
          ret->ShareData(&t);
          return ret;
        }, py::return_value_policy::take_ownership);

  py::class_<TensorList<GPUBackend>>(m, "TensorListGPU", py::buffer_protocol())
    .def("__init__", [](TensorList<GPUBackend> &t) {
          // Construct a default TensorList on GPU
          new (&t) TensorList<GPUBackend>;
        })
    .def("asCPU", [](TensorList<GPUBackend> &t) -> TensorList<CPUBackend>* {
          TensorList<CPUBackend> * ret = new TensorList<CPUBackend>();
          UserStream * us = UserStream::Get();
          cudaStream_t s = us->GetStream(t);
          ret->Copy(t, s);
          CUDA_CALL(cudaStreamSynchronize(s));
          return ret;
        }, py::return_value_policy::take_ownership)
    .def("__len__", [](TensorList<GPUBackend> &t) {
          return t.ntensor();
        })
    .def("is_dense_tensor", &TensorList<GPUBackend>::IsDenseTensor)
    .def("copy_to_external",
        [](TensorList<GPUBackend> &t, py::object p) {
          PyObject *p_ptr = p.ptr();
          PyObject *ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
          void *ptr = PyLong_AsVoidPtr(ptr_as_int);
          CopyToExternalTensor(&t, ptr);
        })
    .def("as_tensor",
        [](TensorList<GPUBackend> &t) -> Tensor<GPUBackend>* {
          Tensor<GPUBackend> * ret = new Tensor<GPUBackend>();
          ret->ShareData(&t);
          return ret;
        }, py::return_value_policy::take_ownership);
}

static vector<string> GetRegisteredCPUOps() {
  return CPUOperatorRegistry::Registry().RegisteredNames();
}

static vector<string> GetRegisteredGPUOps() {
  return GPUOperatorRegistry::Registry().RegisteredNames();
}

static vector<string> GetRegisteredSupportOps() {
  return SupportOperatorRegistry::Registry().RegisteredNames();
}

static const OpSchema &GetSchema(const string &name) {
  return SchemaRegistry::GetSchema(name);
}
#ifdef NDLL_BUILD_PROTO3
typedef ndll::TFRecordParser::FeatureType TFFeatureType;
typedef ndll::TFRecordParser::Feature TFFeature;
typedef TFFeature::Value TFValue;

TFValue ConvertTFRecordDefaultValue(TFFeatureType type, py::object val) {
  PyObject *ptr = val.ptr();
  TFValue ret;
  switch (type) {
    case TFFeatureType::int64:
      NDLL_ENFORCE(PyInt_Check(ptr) || PyLong_Check(ptr),
          "Invalid type for default value, expected int.");
      ret.int64 = PyInt_AsLong(ptr);
      break;
    case TFFeatureType::string:
      NDLL_ENFORCE(PyStr_Check(ptr),
          "Invalid type for default value, expected string.");
      ret.str = PyStr_AsString(ptr);
      break;
    case TFFeatureType::float32:
      NDLL_ENFORCE(PyFloat_Check(ptr),
          "Invalid type for default value, expected float.");
      ret.float32 = PyFloat_AsDouble(ptr);
      break;
    default:
      NDLL_FAIL("Invalid type for default value, expected string, int or float.");
  }
  return ret;
}
#endif  // NDLL_BUILD_PROTO3

PYBIND11_MODULE(ndll_backend, m) {
  m.doc() = "Python bindings for the C++ portions of NDLL";

  // NDLL Init function
  m.def("Init", &NDLLInit);

  // Types
  py::module types_m = m.def_submodule("types");
  types_m.doc() = "Datatypes and options used by NDLL";

  // NDLLDataType
  py::enum_<NDLLDataType>(types_m, "NDLLDataType", "Data type of image")
    .value("NO_TYPE", NDLL_NO_TYPE)
    .value("UINT8", NDLL_UINT8)
    .value("FLOAT16", NDLL_FLOAT16)
    .value("FLOAT", NDLL_FLOAT)
    .value("INT64", NDLL_INT64)
    .value("INT32", NDLL_INT32)
    .value("BOOL", NDLL_BOOL)
    .export_values();

  // NDLLImageType
  py::enum_<NDLLImageType>(types_m, "NDLLImageType", "Image type")
    .value("RGB", NDLL_RGB)
    .value("BGR", NDLL_BGR)
    .value("GRAY", NDLL_GRAY)
    .export_values();

  // NDLLInterpType
  py::enum_<NDLLInterpType>(types_m, "NDLLInterpType", "Interpolation mode")
    .value("INTERP_NN", NDLL_INTERP_NN)
    .value("INTERP_LINEAR", NDLL_INTERP_LINEAR)
    .value("INTERP_CUBIC", NDLL_INTERP_CUBIC)
    .export_values();

  // NDLLTensorLayout
  py::enum_<NDLLTensorLayout>(types_m, "NDLLTensorLayout", "Tensor layout")
    .value("NCHW", NDLL_NCHW)
    .value("NHWC", NDLL_NHWC)
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
            [](int batch_size, int num_threads, int device_id, int seed = -1,
                bool pipelined_execution = true, bool async_execution = true,
                size_t bytes_per_sample_hint = 0, bool set_affinity = false,
                int max_num_stream = -1) {
              return std::unique_ptr<Pipeline>(
                  new Pipeline(batch_size, num_threads, device_id, seed, pipelined_execution,
                      async_execution, bytes_per_sample_hint, set_affinity, max_num_stream));
            }),
        "batch_size"_a,
        "num_threads"_a,
        "device_id"_a,
        "seed"_a,
        "exec_pipelined"_a,
        "exec_async"_a,
        "bytes_per_sample_hint"_a = 0,
        "set_affinity"_a = false,
        "max_num_stream"_a = -1
        )
    // initialize from serialized pipeline
    .def(py::init(
          [](string serialized_pipe,
             int batch_size, int num_threads, int device_id, int seed = -1,
             bool pipelined_execution = true, bool async_execution = true,
             size_t bytes_per_sample_hint = 0, bool set_affinity = false,
             int max_num_stream = -1) {
              return std::unique_ptr<Pipeline>(
                  new Pipeline(serialized_pipe,
                               batch_size, num_threads, device_id, seed, pipelined_execution,
                               async_execution, bytes_per_sample_hint, set_affinity,
                               max_num_stream));
            }),
        "serialized_pipe"_a,
        "batch_size"_a,
        "num_threads"_a,
        "device_id"_a,
        "seed"_a,
        "exec_pipelined"_a,
        "exec_async"_a,
        "bytes_per_sample_hint"_a = 0,
        "set_affinity"_a = false,
        "max_num_stream"_a = -1
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
    .def("RunCPU", &Pipeline::RunCPU)
    .def("RunGPU", &Pipeline::RunGPU)
    .def("Outputs",
        [](Pipeline *p) {
          DeviceWorkspace ws;
          p->Outputs(&ws);

          py::list list;
          for (int i = 0; i < ws.NumOutput(); ++i) {
            if (ws.OutputIsType<CPUBackend>(i)) {
              list.append(ws.Output<CPUBackend>(i));
            } else {
              list.append(ws.Output<GPUBackend>(i));
            }
          }
          return list;
        }, py::return_value_policy::take_ownership)
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
          // issues w/ pybind and a non-copyable type (ndll::Tensor).
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
          NDLL_ENFORCE(sizes.find(op_name) != sizes.end(),
              "Operator " + op_name + " does not expose valid epoch size.");
          return sizes[op_name];
        });

#define NDLL_OPSPEC_ADDARG(T) \
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
    NDLL_OPSPEC_ADDARG(std::string)
    NDLL_OPSPEC_ADDARG(bool)
    NDLL_OPSPEC_ADDARG(int64)
    NDLL_OPSPEC_ADDARG(float)
#ifdef NDLL_BUILD_PROTO3
    NDLL_OPSPEC_ADDARG(TFFeature)
#endif
    .def("AddArg",
        [](OpSpec *spec, const string &name, py::object obj) -> OpSpec& {
          NDLL_FAIL("Unsupported argument type with name " + name);
          return *spec;
        }, py::return_value_policy::reference_internal)
    .def("__repr__", &OpSpec::ToString)
    .def("copy", [](OpSpec &o) -> OpSpec * {
        OpSpec * ret = new OpSpec(o);
        return ret;
        }, py::return_value_policy::reference);

  // Registries for cpu & gpu operators
  m.def("RegisteredCPUOps", &GetRegisteredCPUOps);
  m.def("RegisteredGPUOps", &GetRegisteredGPUOps);
  m.def("RegisteredSupportOps", &GetRegisteredSupportOps);

  // Registry for OpSchema
  m.def("GetSchema", &GetSchema);

  py::class_<OpSchema>(m, "OpSchema")
    .def("Dox", &OpSchema::Dox)
    .def("MaxNumInput", &OpSchema::MaxNumInput)
    .def("MinNumInput", &OpSchema::MinNumInput)
    .def("HasOutputFn", &OpSchema::HasOutputFn)
    .def("CalculateOutputs", &OpSchema::CalculateOutputs)
    .def("SupportsInPlace", &OpSchema::SupportsInPlace)
    .def("CheckArgs", &OpSchema::CheckArgs);

  ExposeTensor(m);
  ExposeTensorList(m);

#ifdef NDLL_BUILD_PROTO3
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
#endif  // NDLL_BUILD_PROTO3
}

}  // namespace python
}  // namespace ndll
