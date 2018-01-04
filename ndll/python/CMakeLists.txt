# Get all the srcs
file(GLOB tmp *.cc)
set(ndll_python_lib "ndll_backend")
pybind11_add_module(${ndll_python_lib} MODULE ${tmp})
add_dependencies(${ndll_python_lib} ${ndll_lib})
target_link_libraries(${ndll_python_lib} PRIVATE ${ndll_lib} ${NDLL_LIBS})
set_target_properties(${ndll_python_lib} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/ndll/python/ndll)

file(COPY "${PROJECT_SOURCE_DIR}/ndll/python" DESTINATION "${PROJECT_BINARY_DIR}/ndll" FILES_MATCHING PATTERN "*.py")

# Add install rule
install(CODE "execute_process(COMMAND pip install --upgrade .
                              WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/ndll/python)")
