#include <pybind11/pybind11.h>

#include "ndll/pipeline/data_reader.h"
#include "ndll/pipeline/data/buffer.h"
#include "ndll/pipeline/pipeline.h"

namespace ndll {
namespace python {

namespace py = pybind11;
using namespace pybind11::literals;

int add(int i, int j) {
  return i + j;
}

PYBIND11_MODULE(ndll_python, m) {
  m.doc() = "This is a test";
  m.def("add", &add, "A function which adds two numbers.");
  
  py::class_<Pipeline>(m, "Pipeline")
    .def(py::init<int, int, int, bool>(), "batch_size"_a,
        "num_threads"_a, "device_id"_a, "set_affinity"_a=true)
    .def("batch_size", &Pipeline::batch_size)
    .def("num_threads", &Pipeline::num_threads)
    .def("AddPrefetchOp", &Pipeline::AddPrefetchOp)
    .def("AddDataReader", &Pipeline::AddDataReader);

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
}

} // namespace python
} // namespace ndll
