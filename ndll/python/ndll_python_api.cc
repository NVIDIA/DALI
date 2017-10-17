#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndll/pipeline/data_reader.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/pipeline/init.h"

namespace ndll {
namespace python {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(pyndll, m) {
  m.doc() = "This is a test";

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
            [](int batch_size, int num_threads, int64 stream_id, int device_id,
                bool set_affinity = true, size_t pixels_per_image_hint = 0) {
              return std::unique_ptr<Pipeline>(
                  new Pipeline(batch_size, num_threads, (cudaStream_t)stream_id,
                      device_id, set_affinity, pixels_per_image_hint)
                  );
            })
        )
    .def("AddDataReader", &Pipeline::AddDataReader)
    .def("AddParser", &Pipeline::AddParser)
    .def("AddDecoder", &Pipeline::AddDecoder)
    .def("AddTransform", &Pipeline::AddTransform)
    .def("Build", &Pipeline::Build)
    .def("RunPrefetch", &Pipeline::RunPrefetch)
    .def("RunCopy", &Pipeline::RunCopy)
    .def("RunForward", &Pipeline::RunForward)
    .def("batch_size", &Pipeline::batch_size)
    .def("num_threads", &Pipeline::num_threads)
    .def("stream_id", [](const Pipeline &pipe) { return (int64)pipe.stream(); });

  py::class_<OpSpec>(m, "OpSpec")
    .def(py::init<std::string, std::string>())
    .def("AddExtraInput", &OpSpec::AddExtraInput,
        py::return_value_policy::reference_internal)
    .def("AddExtraOutput", &OpSpec::AddExtraOutput,
        py::return_value_policy::reference_internal)
    .def("AddExtraGPUInput", &OpSpec::AddExtraGPUInput,
        py::return_value_policy::reference_internal)
    .def("AddExtraGPUOutput", &OpSpec::AddExtraGPUOutput,
        py::return_value_policy::reference_internal)
    .def("AddArg",
        [](OpSpec *spec, const string &name, py::object obj) -> OpSpec& {
          // TODO(tgale): Can we clean this conversion up? Do we want to handle
          // cast errors from pybind so we can give the user better error messages?
          PyObject *value = obj.ptr();
          // Switch on supported data types
          if (PyString_Check(value)) {
            std::string str_val(PyString_AsString(value));
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
            if (PyString_Check(elt)) {
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


  /*
  //
  /// Data Storage
  //

  // Note: We do not expose "SetType" here
  py::class_<TypeMeta>(m, "TypeMeta")
    .def(py::init<>())
    .def("id", &TypeMeta::id)
    .def("size", &TypeMeta::size)
    .def("name", &TypeMeta::name);

  // Note: We do not expose templated mutable_data & data methods
  py::class_<Buffer<CPUBackend>>(m, "BufferCPU")
    .def("raw_mutable_data", &Buffer<CPUBackend>::raw_mutable_data,
        py::return_value_policy::reference_internal)
    .def("raw_data", &Buffer<CPUBackend>::raw_data,
        py::return_value_policy::reference_internal)
    .def("size", &Buffer<CPUBackend>::size)
    .def("nbytes", &Buffer<CPUBackend>::nbytes)
    .def("type", &Buffer<CPUBackend>::type)
    .def("set_type", &Buffer<CPUBackend>::set_type);

  py::class_<Tensor<CPUBackend>, Buffer<CPUBackend>>(m, "TensorCPU")
    .def(py::init<>());
    
  //
  /// Data Readers
  //
  
  py::class_<DataReader>(m, "DataReader")
    .def("Read", &DataReader::Read)
    .def("Reset", &DataReader::Reset)
    .def("Clone", &DataReader::Clone);
    
  py::class_<BatchDataReader, DataReader>(m, "BatchDataReader")
    .def("Read", &BatchDataReader::Read)
    .def("Reset", &BatchDataReader::Reset)
    .def("Clone", &BatchDataReader::Clone);
  */
}

} // namespace python
} // namespace ndll
