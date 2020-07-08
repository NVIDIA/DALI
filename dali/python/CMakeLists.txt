# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Get all the srcs
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_PYTHON_BACKEND_SRCS)

add_custom_target(dali_python ALL)

foreach(PYVER ${PYTHON_VERSIONS})

    # check if listed python versions are accesible
    execute_process(
        COMMAND python${PYVER} -c "print('test', end='')"
        OUTPUT_VARIABLE PYTHON_EXISTS)

    if ("${PYTHON_EXISTS}" STREQUAL "test")
        execute_process(
            COMMAND python${PYVER}-config --extension-suffix
            OUTPUT_VARIABLE PYTHON_SUFIX)
        # remove newline and the extension
        string(REPLACE ".so\n" "" PYTHON_SUFIX ${PYTHON_SUFIX})

        execute_process(
            COMMAND python${PYVER}-config --includes
            OUTPUT_VARIABLE PYTHON_INCLUDES)
        # split and make it a list
        string(REPLACE "-I" "" PYTHON_INCLUDES ${PYTHON_INCLUDES})
        string(REPLACE "\n" "" PYTHON_INCLUDES ${PYTHON_INCLUDES})
        separate_arguments(PYTHON_INCLUDES)

        add_library(dali_python_${PYVER} SHARED ${DALI_PYTHON_BACKEND_SRCS})
        target_link_libraries(dali_python_${PYVER} PUBLIC dali dali_operators dali_kernels dali_core ${CUDART_LIB})
        target_link_libraries(dali_python_${PYVER} PRIVATE ${CUDA_LIBRARIES})
        target_link_libraries(dali_python_${PYVER} PRIVATE "-Wl,--exclude-libs,${exclude_libs}")

        set_target_properties(dali_python_${PYVER}
                              PROPERTIES
                              LIBRARY_OUTPUT_DIRECTORY ${DALI_LIBRARY_OUTPUT_DIR}
                              PREFIX ""
                              OUTPUT_NAME backend_impl${PYTHON_SUFIX})
        target_compile_definitions(dali_python_${PYVER} PRIVATE "-Ddali_python_EXPORTS")
        target_include_directories(dali_python_${PYVER} PRIVATE "${CMAKE_SOURCE_DIR}/third_party/pybind11/include")
        # add includes
        foreach(incl_dir ${PYTHON_INCLUDES})
            target_include_directories(dali_python_${PYVER} PRIVATE ${incl_dir})
        endforeach(incl_dir)

        add_dependencies(dali_python dali_python_${PYVER})
    endif()

endforeach(PYVER)

if (DALI_BUILD_FLAVOR)
  set(DALI_FLAVOR "${DALI_BUILD_FLAVOR} ")
  set(DALI_FLAVOR_MINUS "-${DALI_BUILD_FLAVOR}")
  set(DALI_VERSION "${DALI_VERSION}.${TIMESTAMP}")
endif()

# Add the COPYRIGHT, LICENSE, and Acknowledgements
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/dali/python/nvidia" "${PROJECT_BINARY_DIR}/dali/python")
configure_file("${PROJECT_SOURCE_DIR}/dali/python/__init__.py.in" "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/__init__.py")
configure_file("${PROJECT_SOURCE_DIR}/dali/python/setup.py.in" "${PROJECT_BINARY_DIR}/stage/setup.py")
copy_post_build(dali_python "${PROJECT_BINARY_DIR}/stage/setup.py" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/dali/python/MANIFEST.in" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/tools/rec2idx.py" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/tools/tfrecord2idx" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/Acknowledgements.txt" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/COPYRIGHT" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_BINARY_DIR}/dali/python")
