# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(dali_imgcodec CUDA CXX C)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(decoders)
add_subdirectory(parsers)
add_subdirectory(tools)
add_subdirectory(util)

# Get all the source files and dump test files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_IMGCODEC_SRCS)
collect_test_sources(DALI_IMGCODEC_TEST_SRCS)

add_library(dali_imgcodec ${LIBTYPE} ${DALI_IMGCODEC_SRCS})
target_link_libraries(dali_imgcodec PUBLIC dali_core dali_kernels)

target_link_libraries(dali_imgcodec PRIVATE ${DALI_LIBS} dynlink_cuda)
target_link_libraries(dali_imgcodec PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
set_target_properties(dali_imgcodec PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DALI_LIBRARY_OUTPUT_DIR}")

set(lib_exports "libdali_imgcodec.map")
configure_file("${DALI_ROOT}/cmake/${lib_exports}.in" "${CMAKE_BINARY_DIR}/${lib_exports}")
target_link_libraries(dali_imgcodec PRIVATE  -Wl,--version-script=${CMAKE_BINARY_DIR}/${lib_exports})

if (BUILD_TEST)
  # TODO(janton): create a test_utils_lib with dali_test_config.cc and other common utilities
  adjust_source_file_language_property("${DALI_IMGCODEC_TEST_SRCS}")
  add_executable(dali_imgcodec_test
    ${DALI_IMGCODEC_TEST_SRCS}
    ${DALI_ROOT}/dali/test/dali_test_config.cc)

  # TODO(janton): Remove dependency with target `dali`
  target_link_libraries(dali_imgcodec_test PUBLIC dali_imgcodec dali)
  target_link_libraries(dali_imgcodec_test PRIVATE gtest dynlink_cuda stdc++fs ${DALI_LIBS})
  target_link_libraries(dali_imgcodec_test PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
  target_link_libraries(dali_imgcodec_test PRIVATE "-pie")
  set_target_properties(dali_imgcodec_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dali_imgcodec_test PROPERTIES OUTPUT_NAME "dali_imgcodec_test.bin")
  set_target_properties(dali_imgcodec_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})

  add_check_gtest_target("check-imgcodec-gtest" dali_imgcodec_test ${TEST_BINARY_DIR})
endif()
