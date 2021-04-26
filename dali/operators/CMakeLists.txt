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

project(dali_operator CUDA CXX C)

add_subdirectory(audio)
add_subdirectory(bbox)
add_subdirectory(geometry)
add_subdirectory(debug)
add_subdirectory(decoder)
add_subdirectory(generic)
add_subdirectory(image)
add_subdirectory(math)
add_subdirectory(random)
add_subdirectory(reader)
add_subdirectory(segmentation)
add_subdirectory(sequence)
add_subdirectory(signal)
add_subdirectory(ssd)
add_subdirectory(util)
add_subdirectory(numba_function)
if (BUILD_PYTHON)
  add_subdirectory(python_function)
endif()

# Get all the source files and dump test files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_OPERATOR_SRCS)
collect_test_sources(DALI_OPERATOR_TEST_SRCS)

set(DALI_OPERATOR_OBJ)
if (BUILD_LMDB)
  list(APPEND DALI_OPERATOR_OBJ $<TARGET_OBJECTS:CAFFE_PROTO> $<TARGET_OBJECTS:CAFFE2_PROTO>)
endif()
if (BUILD_PROTO3)
  list(APPEND DALI_OPERATOR_OBJ $<TARGET_OBJECTS:TF_PROTO>)
endif()

if (BUILD_NVDEC)
  list(APPEND DALI_OPERATOR_OBJ $<TARGET_OBJECTS:NVCUVID_GEN>)
endif(BUILD_NVDEC)


adjust_source_file_language_property("${DALI_OPERATOR_SRCS}")
add_library(dali_operators ${LIBTYPE} ${DALI_OPERATOR_SRCS} ${DALI_OPERATOR_OBJ})
set_target_properties(dali_operators PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DALI_LIBRARY_OUTPUT_DIR}")
target_link_libraries(dali_operators PUBLIC dali dali_kernels dali_core)
target_link_libraries(dali_operators PRIVATE dynlink_cuda ${DALI_LIBS})
if (BUILD_NVML)
  target_link_libraries(dali_operators PRIVATE $<TARGET_OBJECTS:dynlink_nvml>)
endif(BUILD_NVML)

if (BUILD_CUFILE)
  target_link_libraries(dali_operators PRIVATE dynlink_cufile)
endif()
# Exclude (most) statically linked dali dependencies from the exports of libdali_operators.so
target_link_libraries(dali_operators PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
# Options for using Dockerfile FFmpeg version
target_compile_definitions(dali_operators PUBLIC HAVE_AVSTREAM_CODECPAR=1)
target_compile_definitions(dali_operators PUBLIC HAVE_AVBSFCONTEXT=1)

set(lib_exports "libdali_operators.map")
configure_file("${DALI_ROOT}/cmake/${lib_exports}.in" "${CMAKE_BINARY_DIR}/${lib_exports}")
target_link_libraries(dali_operators PRIVATE -Wl,--version-script=${CMAKE_BINARY_DIR}/${lib_exports})

if (BUILD_TEST)
  # TODO(janton): create a test_utils_lib with dali_test_config.cc and other common utilities
  adjust_source_file_language_property("${DALI_OPERATOR_TEST_SRCS}")
  add_executable(dali_operator_test
    ${DALI_OPERATOR_TEST_SRCS}
    ${DALI_ROOT}/dali/test/dali_test_config.cc
    ${DALI_ROOT}/dali/test/dali_operator_test_utils.cc)

  target_link_libraries(dali_operator_test PUBLIC dali_operators)
  target_link_libraries(dali_operator_test PRIVATE gtest dynlink_cuda ${DALI_LIBS})
  if (BUILD_NVML)
    target_link_libraries(dali_operator_test PRIVATE $<TARGET_OBJECTS:dynlink_nvml>)
  endif(BUILD_NVML)
  if (BUILD_CUFILE)
    target_link_libraries(dali_operator_test PRIVATE dynlink_cufile)
  endif()
  target_link_libraries(dali_operator_test PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
  target_link_libraries(dali_operator_test PRIVATE "-pie")
  set_target_properties(dali_operator_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dali_operator_test PROPERTIES OUTPUT_NAME "dali_operator_test.bin")
  set_target_properties(dali_operator_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})

  add_check_gtest_target("check-operator-gtest" dali_operator_test ${TEST_BINARY_DIR})
endif()
