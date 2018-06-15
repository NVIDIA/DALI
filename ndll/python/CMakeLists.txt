# Get all the srcs
file(GLOB tmp *.cc)
set(dali_python_lib "backend_impl")
pybind11_add_module(${dali_python_lib} MODULE ${tmp})
add_dependencies(${dali_python_lib} ${ndll_lib})
if (BUILD_TENSORFLOW)
  add_dependencies(${dali_python_lib} ${customop_lib})
endif()
target_link_libraries(${dali_python_lib} PRIVATE ${ndll_lib} ${NDLL_LIBS})
set_target_properties(${dali_python_lib} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/ndll/python/dali)

# Add install rule
install(DIRECTORY "${PROJECT_SOURCE_DIR}/ndll/python/dali" DESTINATION "${PROJECT_BINARY_DIR}/ndll/python")
install(FILES "${PROJECT_SOURCE_DIR}/ndll/python/setup.py" DESTINATION "${PROJECT_BINARY_DIR}/ndll/python")
install(FILES "${PROJECT_SOURCE_DIR}/tools/rec2idx.py" DESTINATION "${PROJECT_BINARY_DIR}/ndll/python")
install(FILES "${PROJECT_SOURCE_DIR}/tools/tfrecord2idx" DESTINATION "${PROJECT_BINARY_DIR}/ndll/python")

# Get the COPYRIGHT, LICENSE, and Acknowledgements
install(FILES "${PROJECT_SOURCE_DIR}/Acknowledgements.txt" DESTINATION "${PROJECT_BINARY_DIR}/ndll/python/dali")
install(FILES "${PROJECT_SOURCE_DIR}/COPYRIGHT" DESTINATION "${PROJECT_BINARY_DIR}/ndll/python/dali")
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" DESTINATION "${PROJECT_BINARY_DIR}/ndll/python/dali")
