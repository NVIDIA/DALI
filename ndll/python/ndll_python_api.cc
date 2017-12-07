#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndll/pipeline/pipeline.h"
#include "ndll/pipeline/init.h"

namespace ndll {
namespace python {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(ndll_cxx, m) {
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
            [](int batch_size, int num_threads, int device_id, int queue_depth = 2,
                size_t bytes_per_sample_hint = 0, bool set_affinity = false,
                int max_num_stream = -1) {
              return std::unique_ptr<Pipeline>(
                  new Pipeline(batch_size, num_threads, device_id, queue_depth,
                      bytes_per_sample_hint, set_affinity, max_num_stream)
                  );
            }),
        "batch_size"_a,
        "num_threads"_a,
        "device_id"_a,
        "queue_depth"_a = 2,
        "bytes_per_sample_hint"_a = 0,
        "set_affinity"_a = false,
        "max_num_stream"_a = -1
        )
    .def("AddOperator", &Pipeline::AddOperator)
    .def("Build", &Pipeline::Build)
    .def("RunCPU", &Pipeline::RunCPU)
    .def("RunGPU", &Pipeline::RunGPU)
    .def("batch_size", &Pipeline::batch_size)
    .def("num_threads", &Pipeline::num_threads);


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
}

} // namespace python
} // namespace ndll
