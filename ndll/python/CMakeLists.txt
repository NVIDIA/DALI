# Get all the srcs
file(GLOB tmp *.cc)
set(ndll_python_lib "ndll_python")
pybind11_add_module(${ndll_python_lib} MODULE ${tmp})
add_dependencies(${ndll_python_lib} ${ndll_lib})
set(test -Wl,--whole-archive ${ndll_lib} -Wl,--no-whole-archive)
target_link_libraries(${ndll_python_lib} PRIVATE ${test} ${NDLL_LIBS})
