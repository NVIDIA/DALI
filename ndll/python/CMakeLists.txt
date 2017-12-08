# Get all the srcs
file(GLOB tmp *.cc)
set(ndll_python_lib "ndll_backend")
pybind11_add_module(${ndll_python_lib} MODULE ${tmp})
add_dependencies(${ndll_python_lib} ${ndll_lib})
target_link_libraries(${ndll_python_lib} PRIVATE ${ndll_lib} ${NDLL_LIBS})
set_target_properties(${ndll_python_lib} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/ndll)

# Copy all python files to the binary dir
file(GLOB tmp *.py)
foreach (ITEM ${tmp})
  get_filename_component(FILE ${ITEM} NAME)
  configure_file(${ITEM} ${PROJECT_BINARY_DIR}/ndll/${FILE} COPYONLY)
endforeach()

# Add install rule
install(TARGETS ${ndll_python_lib} DESTINATION lib)
