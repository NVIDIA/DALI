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

project(dali_kernels CUDA CXX C)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(audio)
add_subdirectory(common)
add_subdirectory(erase)
add_subdirectory(imgproc)
add_subdirectory(math)
add_subdirectory(normalize)
add_subdirectory(reduce)
add_subdirectory(signal)
add_subdirectory(slice)
add_subdirectory(test)
add_subdirectory(transpose)

# Get all the source files and dump test files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_KERNEL_SRCS)
collect_test_sources(DALI_KERNEL_TEST_SRCS)

# cuFFT library
adjust_source_file_language_property("${DALI_KERNEL_SRCS}")
set_source_files_properties("dummy.cu" PROPERTIES LANGUAGE CUDA)
add_library(dali_kernels ${LIBTYPE} ${DALI_KERNEL_SRCS})
target_link_libraries(dali_kernels PUBLIC dali_core)
target_link_libraries(dali_kernels PRIVATE ${CUDA_cufft_static_LIBRARY})
target_link_libraries(dali_kernels PRIVATE ${DALI_LIBS} dynlink_cuda)
target_link_libraries(dali_kernels PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
set_target_properties(dali_kernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DALI_LIBRARY_OUTPUT_DIR}")

set(lib_exports "libdali_kernels.map")
configure_file("${DALI_ROOT}/cmake/${lib_exports}.in" "${CMAKE_BINARY_DIR}/${lib_exports}")
target_link_libraries(dali_kernels PRIVATE  -Wl,--version-script=${CMAKE_BINARY_DIR}/${lib_exports})

if (BUILD_TEST)
  # TODO(janton): create a test_utils_lib with dali_test_config.cc and other common utilities
  adjust_source_file_language_property("${DALI_KERNEL_TEST_SRCS}")
  add_executable(dali_kernel_test
    ${DALI_KERNEL_TEST_SRCS}
    ${DALI_ROOT}/dali/test/dali_test_config.cc)

  # TODO(janton): Remove dependency with target `dali`
  target_link_libraries(dali_kernel_test PUBLIC dali_kernels dali)
  target_link_libraries(dali_kernel_test PRIVATE gtest dynlink_cuda ${DALI_LIBS})
  target_link_libraries(dali_kernel_test PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
  target_link_libraries(dali_kernel_test PRIVATE "-pie")
  set_target_properties(dali_kernel_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dali_kernel_test PROPERTIES OUTPUT_NAME "dali_kernel_test.bin")
  set_target_properties(dali_kernel_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})

  add_check_gtest_target("check-kernel-gtest" dali_kernel_test ${TEST_BINARY_DIR})
endif()
