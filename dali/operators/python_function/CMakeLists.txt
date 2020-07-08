# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(PYTHON_FUNCTION_SRCS PARENT_SCOPE)
collect_test_sources(DALI_OPERATOR_TEST_SRCS PARENT_SCOPE)

add_custom_target(${dali_python_function_lib} ALL)

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

        add_library(${dali_python_function_lib}_${PYVER} SHARED ${PYTHON_FUNCTION_SRCS})
        target_link_libraries(${dali_python_function_lib}_${PYVER} PUBLIC dali)
        target_link_libraries(${dali_python_function_lib}_${PYVER} PRIVATE ${DALI_LIBS})
        target_link_libraries(${dali_python_function_lib}_${PYVER} PRIVATE "-Wl,--exclude-libs,${exclude_libs}")

        set_target_properties(${dali_python_function_lib}_${PYVER}
                                PROPERTIES
                                LIBRARY_OUTPUT_DIRECTORY ${DALI_LIBRARY_OUTPUT_DIR}
                                PREFIX ""
                                OUTPUT_NAME ${dali_python_function_lib}${PYTHON_SUFIX})
        target_include_directories(${dali_python_function_lib}_${PYVER} PRIVATE "${CMAKE_SOURCE_DIR}/third_party/pybind11/include")
        # add includes
        foreach(incl_dir ${PYTHON_INCLUDES})
            target_include_directories(${dali_python_function_lib}_${PYVER} PRIVATE ${incl_dir})
        endforeach(incl_dir)

        add_dependencies(${dali_python_function_lib} ${dali_python_function_lib}_${PYVER})
    endif()

endforeach(PYVER)
