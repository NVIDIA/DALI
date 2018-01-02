// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndll/pipeline/init.h"
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/op_schema.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"
#include "ndll/python/python3_compat.h"

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
  } else if (IsType<long long>(type)) { // NOLINT
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
    return TypeInfo::Create<long long>(); // NOLINT
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

void ExposeTensorCPU(py::module &m) { // NOLINT
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
    .def("resize", &Tensor<CPUBackend>::Resize);
}

void ExposeTensorListCPU(py::module &m) { // NOLINT
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
        });

  py::class_<TensorList<GPUBackend>>(m, "TensorListGPU", py::buffer_protocol())
    .def("__init__", [](TensorList<GPUBackend> &t) {
          // Construct a default TensorList on GPU
          new (&t) TensorList<GPUBackend>;
        });
}

static vector<string> GetRegisteredCPUOps() {
  return CPUOperatorRegistry::Registry().RegisteredNames();
}

static vector<string> GetRegisteredGPUOps() {
  return GPUOperatorRegistry::Registry().RegisteredNames();
}

static OpSchema GetSchema(const string &name) {
  return SchemaRegistry::GetSchema(name);
}

PYBIND11_MODULE(ndll_backend, m) {
  m.doc() = "Python bindings for the C++ portions of NDLL";

  // NDLL Init function
  m.def("Init", &NDLLInit);

  // NDLLDataType, NDLLImageType, NDLLInterpType enums
  m.attr("NO_TYPE") = -1;
  m.attr("UINT8") = 0;
  m.attr("FLOAT16") = 1;
  m.attr("FLOAT") = 2;

  m.attr("RGB") = 0;
  m.attr("BGR") = 1;
  m.attr("GRAY") = 2;

  m.attr("INTERP_NN") = 0;
  m.attr("INTERP_LINEAR") = 1;
  m.attr("INTERP_CUBIC") = 2;

  // Pipeline class
  py::class_<Pipeline>(m, "Pipeline")
    .def(py::init(
            [](int batch_size, int num_threads, int device_id,
                bool pipelined_execution = false, bool async_execution = false,
                size_t bytes_per_sample_hint = 0, bool set_affinity = false,
                int max_num_stream = -1) {
              return std::unique_ptr<Pipeline>(
                  new Pipeline(batch_size, num_threads, device_id, pipelined_execution,
                      async_execution, bytes_per_sample_hint, set_affinity, max_num_stream));
            }),
        "batch_size"_a,
        "num_threads"_a,
        "device_id"_a,
        "exec_pipelined"_a,
        "exec_async"_a,
        "bytes_per_sample_hint"_a = 0,
        "set_affinity"_a = false,
        "max_num_stream"_a = -1
        )
    .def("AddOperator", &Pipeline::AddOperator)
    .def("Build", &Pipeline::Build)
    .def("RunCPU", &Pipeline::RunCPU)
    .def("RunGPU", &Pipeline::RunGPU)
    .def("Outputs",
        [](Pipeline *p) {
          DeviceWorkspace ws;
          p->Outputs(&ws);

          py::list list;
          for (int i = 0; i < ws.NumOutput(); ++i) {
            if (ws.OutputIsType<CPUBackend>(0)) {
              list.append(ws.Output<CPUBackend>(0));
            } else {
              list.append(ws.Output<GPUBackend>(0));
            }
          }
          return list;
        }, py::return_value_policy::take_ownership)
    .def("batch_size", &Pipeline::batch_size)
    .def("num_threads", &Pipeline::num_threads)
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
        });

  py::class_<OpSpec>(m, "OpSpec")
    .def(py::init<std::string>(), "name"_a)
    .def("AddInput", &OpSpec::AddInput,
        py::return_value_policy::reference_internal)
    .def("AddOutput", &OpSpec::AddOutput,
        py::return_value_policy::reference_internal)
    .def("AddArg",
        [](OpSpec *spec, const string &name, py::object obj) -> OpSpec& {
          // TODO(tgale): Can we clean this conversion up? Do we want to handle
          // cast errors from pybind so we can give the user better error messages?
          PyObject *value = obj.ptr();
          // Switch on supported data types
          if (PyStr_Check(value)) {
            std::string str_val(PyStr_AsString(value));
            spec->AddArg(name, str_val);
          } else if (PyBool_Check(value)) {
            bool bool_val(value == Py_True);
            spec->AddArg(name, bool_val);
          } else if (PyInt_Check(value) || PyLong_Check(value)) {
            int64 int_val = PyInt_AsLong(value);
            spec->AddArg(name, int_val);
          } else if (PyFloat_Check(value)) {
            double float_val = PyFloat_AsDouble(value);
            spec->AddArg(name, float_val);
          } else if (PyList_Check(value)) {
            size_t size = PyList_Size(value);
            NDLL_ENFORCE(size > 0, "Empty list arguments not supported.");

            // Get the first type
            PyObject *elt = PyList_GetItem(value, 0);
            if (PyStr_Check(elt)) {
              vector<string> str_vals = obj.cast<vector<string>>();
              spec->AddArg(name, str_vals);
            } else if (PyBool_Check(elt)) {
              vector<bool> bool_vals = obj.cast<vector<bool>>();
              spec->AddArg(name, bool_vals);
            } else if (PyInt_Check(elt) || PyLong_Check(value)) {
              vector<int64> int_vals = obj.cast<vector<int64>>();
              spec->AddArg(name, int_vals);
            } else if (PyFloat_Check(elt)) {
              vector<double> float_vals = obj.cast<vector<double>>();
              spec->AddArg(name, float_vals);
            } else {
              NDLL_FAIL("Unsupported list element type in argument "
                  "with name " + name);
            }
          } else {
            NDLL_FAIL("Unsupported argument type with name " + name);
          }
          return *spec;
        }, py::return_value_policy::reference_internal);

  // Registries for cpu & gpu operators
  m.def("RegisteredCPUOps", &GetRegisteredCPUOps);
  m.def("RegisteredGPUOps", &GetRegisteredGPUOps);

  // Registry for OpSchema
  m.def("GetSchema", &GetSchema);

  py::class_<OpSchema>(m, "OpSchema")
    .def("Dox", &OpSchema::Dox)
    .def("MaxNumInput", &OpSchema::MaxNumInput)
    .def("MinNumInput", &OpSchema::MinNumInput)
    .def("MaxNumOutput", &OpSchema::MaxNumOutput)
    .def("MinNumOutput", &OpSchema::MinNumOutput)
    .def("HasOutputFn", &OpSchema::HasOutputFn)
    .def("CalculateOutputs", &OpSchema::CalculateOutputs)
    .def("SupportsInPlace", &OpSchema::SupportsInPlace);

  ExposeTensorCPU(m);
  ExposeTensorListCPU(m);
}

}  // namespace python
}  // namespace ndll
