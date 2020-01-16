# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

project(dali_kernel)

add_subdirectory(audio)
add_subdirectory(common)
add_subdirectory(erase)
add_subdirectory(imgproc)
add_subdirectory(normalize)
add_subdirectory(reduce)
add_subdirectory(signal)
add_subdirectory(slice)
add_subdirectory(test)

# Get all the source files and dump test files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_KERNEL_SRCS)
collect_test_sources(DALI_KERNEL_TEST_SRCS)

cuda_add_library(${dali_kernel_lib} SHARED ${DALI_KERNEL_SRCS})
set_target_properties(${dali_kernel_lib} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DALI_LIBRARY_OUTPUT_DIR}")
target_link_libraries(${dali_kernel_lib} PUBLIC ${dali_core_lib})
target_link_libraries(${dali_kernel_lib} PRIVATE ${DALI_LIBS})
target_link_libraries(${dali_kernel_lib} PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
set_target_properties(${dali_kernel_lib} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DALI_LIBRARY_OUTPUT_DIR}")

set(lib_exports "libdali_kernels.map")
configure_file("${DALI_ROOT}/cmake/${lib_exports}.in" "${CMAKE_BINARY_DIR}/${lib_exports}")
target_link_libraries(${dali_kernel_lib} PRIVATE -Wl,--version-script=${CMAKE_BINARY_DIR}/${lib_exports})

if (BUILD_TEST)
  set(test_bin "dali_kernel_test.bin")
  # TODO(janton): create a test_utils_lib with dali_test_config.cc and other common utilities
  cuda_add_executable(${test_bin}
    ${DALI_KERNEL_TEST_SRCS}
    ${DALI_ROOT}/dali/test/dali_test_config.cc)

  # TODO(janton): Remove dependency with dali_lib
  target_link_libraries(${test_bin} PUBLIC ${dali_kernel_lib} ${dali_lib})
  target_link_libraries(${test_bin} PRIVATE gtest dynlink_cuda ${DALI_LIBS})
  target_link_libraries(${test_bin} PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
  set_target_properties(${test_bin} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})

  add_check_gtest_target("check-kernel-gtest" ${test_bin} ${TEST_BINARY_DIR})
endif()
