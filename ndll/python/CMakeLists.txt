# Get all the srcs
file(GLOB tmp *.cc)
set(ndll_python_lib "ndll_cxx")
pybind11_add_module(${ndll_python_lib} MODULE ${tmp})
add_dependencies(${ndll_python_lib} ${ndll_lib})
target_link_libraries(${ndll_python_lib} PRIVATE ${ndll_lib} ${NDLL_LIBS})

# Add install rule
install(TARGETS ${ndll_python_lib} DESTINATION lib)
