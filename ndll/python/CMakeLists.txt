# Get all the srcs
file(GLOB tmp *.cc)
pybind11_add_module(ndll_python MODULE ${tmp})
