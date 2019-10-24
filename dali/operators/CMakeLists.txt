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

project(dali_operator)

add_subdirectory(color)
add_subdirectory(color_space)
add_subdirectory(crop)
add_subdirectory(decoder)
add_subdirectory(detection)
add_subdirectory(displacement)
add_subdirectory(expressions)
add_subdirectory(fused)
add_subdirectory(geometric)
if (BUILD_NVOF)
  add_subdirectory(optical_flow)
endif()
add_subdirectory(paste)
add_subdirectory(reader)
add_subdirectory(resize)
add_subdirectory(sequence)
add_subdirectory(support)
add_subdirectory(transpose)
add_subdirectory(util)
if (BUILD_PYTHON)
  add_subdirectory(python_function)
endif()

# Get all the source files and dump test files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_OPERATOR_SRCS)
collect_test_sources(DALI_OPERATOR_TEST_SRCS)

set(DALI_OPERATOR_PROTO_OBJ)
if (BUILD_LMDB)
  list(APPEND DALI_OPERATOR_PROTO_OBJ $<TARGET_OBJECTS:CAFFE_PROTO> $<TARGET_OBJECTS:CAFFE2_PROTO>)
endif()
if (BUILD_PROTO3)
  list(APPEND DALI_OPERATOR_PROTO_OBJ $<TARGET_OBJECTS:TF_PROTO>)
endif()

cuda_add_library(${dali_operator_lib} SHARED ${DALI_OPERATOR_SRCS} ${DALI_OPERATOR_PROTO_OBJ})
set_target_properties(${dali_operator_lib} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DALI_LIBRARY_OUTPUT_DIR}")
target_link_libraries(${dali_operator_lib} PUBLIC ${dali_lib} ${dali_kernel_lib} ${dali_core_lib})
target_link_libraries(${dali_operator_lib} PRIVATE dynlink_cuda ${DALI_LIBS})
# Exclude (most) statically linked dali dependencies from the exports of libdali_operators.so
target_link_libraries(${dali_operator_lib} PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
# Options for using Dockerfile FFmpeg version
target_compile_definitions(${dali_operator_lib} PUBLIC HAVE_AVSTREAM_CODECPAR=1)
target_compile_definitions(${dali_operator_lib} PUBLIC HAVE_AVBSFCONTEXT=1)

set(lib_exports "libdali_operators.map")
configure_file("${DALI_ROOT}/cmake/${lib_exports}.in" "${CMAKE_BINARY_DIR}/${lib_exports}")
target_link_libraries(${dali_operator_lib} PRIVATE -Wl,--version-script=${CMAKE_BINARY_DIR}/${lib_exports})

if (BUILD_TEST)
  set(test_bin "dali_operator_test.bin")
  # TODO(janton): create a test_utils_lib with dali_test_config.cc and other common utilities
  cuda_add_executable(${test_bin}
    ${DALI_OPERATOR_TEST_SRCS}
    ${DALI_ROOT}/dali/test/dali_test_config.cc
    ${DALI_ROOT}/dali/test/dali_operator_test_utils.cc)

  target_link_libraries(${test_bin} PUBLIC ${dali_operator_lib})
  target_link_libraries(${test_bin} PRIVATE gtest dynlink_cuda ${DALI_LIBS})
  target_link_libraries(${test_bin} PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
  set_target_properties(${test_bin} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})

  add_check_gtest_target("check-operator-gtest" ${test_bin} ${TEST_BINARY_DIR})
endif()
