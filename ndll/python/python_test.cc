#include <pybind11/pybind11.h>

int add(int i, int j) {
  return i + j;
}

PYBIND11_MODULE(ndll_python, m) {
  m.doc() = "This is a test";
  m.def("add", &add, "A function which adds two numbers.");
}
