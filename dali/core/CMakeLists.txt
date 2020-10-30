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

project(dali_core CUDA CXX C)

# Get all the source files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_CORE_SRCS)
collect_test_sources(DALI_CORE_TEST_SRCS)

set(CUDA_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_cuda_gen.cc")
add_custom_command(
    OUTPUT ${CUDA_GENERATED_STUB}
    COMMAND python ${CMAKE_SOURCE_DIR}/tools/stub_generator/stub_codegen.py --
                "${CMAKE_SOURCE_DIR}/tools/stub_generator/cuda.json" ${CUDA_GENERATED_STUB}
                "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cuda.h" "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    DEPENDS ${CMAKE_SOURCE_DIR}/tools/stub_generator/stub_codegen.py
    COMMENT "Running cuda.h stub generator"
    VERBATIM )

set_source_files_properties(${CUDA_GENERATED_STUB} PROPERTIES GENERATED TRUE)

list(REMOVE_ITEM DALI_CORE_SRCS dynlink_cuda.cc)

add_library(dynlink_cuda STATIC dynlink_cuda.cc ${CUDA_GENERATED_STUB})

add_library(dali_core ${LIBTYPE} ${DALI_CORE_SRCS})
target_include_directories(dali_core PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(dali_core PRIVATE ${CUDART_LIB} dynlink_cuda)
target_link_libraries(dali_core PUBLIC ${DALI_SYSTEM_LIBS})
target_link_libraries(dali_core PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
set_target_properties(dali_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DALI_LIBRARY_OUTPUT_DIR}")

set(lib_exports "libdali_core.map")
configure_file("${DALI_ROOT}/cmake/${lib_exports}.in" "${CMAKE_BINARY_DIR}/${lib_exports}")
target_link_libraries(dali_core PRIVATE -Wl,--version-script=${CMAKE_BINARY_DIR}/${lib_exports})

if (BUILD_TEST)
  add_executable(dali_core_test "${DALI_CORE_TEST_SRCS}")
  target_link_libraries(dali_core_test PUBLIC dali_core)
  target_link_libraries(dali_core_test PRIVATE gtest dynlink_cuda ${DALI_LIBS})
  target_link_libraries(dali_core_test PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
  target_link_libraries(dali_core_test PRIVATE "-pie")
  set_target_properties(dali_core_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dali_core_test PROPERTIES OUTPUT_NAME "dali_core_test.bin")

  set_target_properties(dali_core_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})

  add_check_gtest_target("check-core-gtest" dali_core_test ${TEST_BINARY_DIR})
endif()
