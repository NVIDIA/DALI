# Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Set variables used by subdirectories
set(DALI_SRCS)
set(DALI_OPERATOR_SRCS)
set(DALI_TEST_SRCS)
set(DALI_BENCHMARK_SRCS)
set(DALI_TF_SRCS)

set(dali_python_function_lib "python_function_plugin")
set(DALI_WHEEL_DIR "dali/python/nvidia/dali")
set(DALI_INCLUDE_DIR "${DALI_WHEEL_DIR}/include/")
set(DALI_LIBRARY_OUTPUT_DIR "${PROJECT_BINARY_DIR}/${DALI_WHEEL_DIR}")
set(TEST_BINARY_DIR "${PROJECT_BINARY_DIR}/${DALI_WHEEL_DIR}/test")

################################################
# Common config for all subdirs
################################################
string(REPLACE ";" ":" exclude_libs "${DALI_EXCLUDES}")

################################################
# Build DALI libraries
################################################
add_subdirectory(core)

add_subdirectory(npp)

if (BUILD_DALI_KERNELS)
  add_subdirectory(kernels)
endif()

if (BUILD_DALI_PIPELINE)
  add_subdirectory(pipeline)
  add_subdirectory(util)
  add_subdirectory(plugin)
  add_subdirectory(c_api)
  add_subdirectory(c_api_2)
endif()

if(BUILD_DALI_OPERATORS)
  if(BUILD_NVJPEG)
    add_subdirectory(nvjpeg)
  endif()

  if(BUILD_NVIMAGECODEC)
    add_subdirectory(nvimgcodec)
  endif()

  add_subdirectory(operators)
endif()

# Collect source files for dali
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_SRCS PARENT_SCOPE)
if (BUILD_PROTOBUF)
  set(DALI_PROTO_OBJ $<TARGET_OBJECTS:DALI_PROTO>)
  adjust_source_file_language_property("${DALI_SRCS}")
  add_library(dali ${LIBTYPE} ${DALI_SRCS} ${DALI_PROTO_OBJ} ${CUDART_LIB})
  set_target_properties(dali PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${DALI_LIBRARY_OUTPUT_DIR}")
endif()

if (BUILD_DALI_PIPELINE)
  # Define symbol version script for libdali.so
  set(dali_lib_exports "libdali.map")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/${dali_lib_exports}.in" "${CMAKE_BINARY_DIR}/${dali_lib_exports}")
  target_link_libraries(dali PRIVATE -Wl,--version-script=${CMAKE_BINARY_DIR}/${dali_lib_exports})

  # Link in dali's dependencies
  message(STATUS "Adding dependencies to target `dali`: '${DALI_LIBS}'")
  target_link_libraries(dali PUBLIC dali_core dali_kernels)
  target_link_libraries(dali PRIVATE ${DALI_LIBS} dynlink_cuda)
  # Exclude (most) statically linked dali dependencies from the exports of libdali.so
  target_link_libraries(dali PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
endif()

if (BUILD_NVML)
  target_link_libraries(dali PRIVATE dynlink_nvml)
endif(BUILD_NVML)

if (BUILD_CUFILE)
  target_link_libraries(dali PRIVATE dynlink_cufile)
endif()

if (BUILD_AWSSDK)
  target_include_directories(dali PRIVATE ${AWSSDK_INCLUDE_DIR})
  target_link_libraries(dali PRIVATE ${AWSSDK_LIBRARIES})
endif()

# Build test suite
################################################
if (BUILD_DALI_PIPELINE AND BUILD_TEST)
  add_subdirectory(test)
  adjust_source_file_language_property("${DALI_TEST_SRCS}")
  add_executable(dali_test "${DALI_TEST_SRCS}")

  target_link_libraries(dali_test PUBLIC dali dali_core dali_kernels ${DALI_LIBS} gtest)
  target_link_libraries(dali_test PRIVATE dynlink_cuda ${CUDART_LIB})
  if (BUILD_NVML)
    target_link_libraries(dali_test PRIVATE dynlink_nvml)
  endif(BUILD_NVML)
  if (BUILD_CUFILE)
    target_link_libraries(dali_test PRIVATE dynlink_cufile)
  endif()
  target_link_libraries(dali_test PRIVATE "-pie")
  set_target_properties(dali_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})
  set_target_properties(dali_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dali_test PROPERTIES OUTPUT_NAME "dali_test.bin")

  add_check_gtest_target("check-dali-gtest" dali_test ${TEST_BINARY_DIR})
endif()


################################################
# Build benchmark suite
################################################
if (BUILD_BENCHMARK)
  # get benchmark main
  add_subdirectory(benchmark)
endif()


################################################
# Build fuzzing suite
################################################
if (BUILD_FUZZING)
  add_subdirectory(fuzzing)
endif()

################################################
# Build the DALI python bindings
################################################
if (BUILD_PYTHON)
  # Get all python srcs
  add_subdirectory(python)

  # prepare check-python target
  add_custom_target(check-python)
  add_dependencies(check check-python)

  set(PYTHON_TARGET_PATH ${PROJECT_BINARY_DIR}/dali/python)
  if($ENV{PYTHONPATH})
    set(PYTHONPATH "${PYTHON_TARGET_PATH}:$ENV{PYTHONPATH}")
  else()
    set(PYTHONPATH "${PYTHON_TARGET_PATH}")
  endif()

  add_dependencies(check-python dali_python)
  add_dependencies(check-python dali)
endif()


################################################
# Gather DALI headers for whl
################################################

# Copy all headers from DALI_INST_HDRS list to DALI_WHEEL_DIR using install command
# with `-D` option, that recursively creates missing directories in destination path
if (BUILD_PYTHON)
  if (PREBUILD_DALI_LIBS)
    add_custom_target(install_headers ALL)
  else (PREBUILD_DALI_LIBS)
    add_custom_target(install_headers ALL
        DEPENDS dali dali_operators
        )
  endif (PREBUILD_DALI_LIBS)

  # Process the DALI_INST_HDRS list
  foreach(INSTALL_HEADER ${DALI_INST_HDRS})
    file(RELATIVE_PATH HEADER_RELATIVE ${PROJECT_SOURCE_DIR} ${INSTALL_HEADER})
    add_custom_command(
      TARGET install_headers
      POST_BUILD
      COMMAND install -D "${INSTALL_HEADER}" "${PROJECT_BINARY_DIR}/${DALI_INCLUDE_DIR}/${HEADER_RELATIVE}")
  endforeach(INSTALL_HEADER)

  # Copy proper `include` dir
  add_custom_command(
    TARGET install_headers
    POST_BUILD
    COMMAND cp -r "${PROJECT_SOURCE_DIR}/include/." "${PROJECT_BINARY_DIR}/${DALI_INCLUDE_DIR}"
  )

  # Copy boost/preprocessor include files
  add_custom_command(
    TARGET install_headers
    POST_BUILD
    COMMAND cp -rL "${PROJECT_SOURCE_DIR}/third_party/boost/preprocessor/include/." "${PROJECT_BINARY_DIR}/${DALI_INCLUDE_DIR}/"
  )
endif()
