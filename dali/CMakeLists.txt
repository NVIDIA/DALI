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

# Set variables used by subdirectories
set(DALI_SRCS)
set(DALI_OPERATOR_SRCS)
set(DALI_TEST_SRCS)
set(DALI_BENCHMARK_SRCS)
set(DALI_TF_SRCS)

set(dali_lib "dali")
set(dali_operator_lib "dali_operators")
set(dali_kernel_lib "dali_kernels")
set(dali_kernel_test_lib "dali_kernel_test")

################################################
# Build libdali
################################################
add_subdirectory(image)
add_subdirectory(kernels)
add_subdirectory(pipeline)
add_subdirectory(util)
add_subdirectory(plugin)
add_subdirectory(c_api)
# Collect source files for dali
file(GLOB tmp *.cc)
set(DALI_SRCS ${DALI_SRCS} ${tmp})

set(DALI_PROTO_OBJ $<TARGET_OBJECTS:DALI_PROTO>)
if (BUILD_LMDB)
  list(APPEND DALI_PROTO_OBJ $<TARGET_OBJECTS:CAFFE_PROTO> $<TARGET_OBJECTS:CAFFE2_PROTO>)
endif()
if (BUILD_PROTO3)
  list(APPEND DALI_PROTO_OBJ $<TARGET_OBJECTS:TF_PROTO>)
endif()

set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
  # In cmake 3.8.0 or lower, CUDA_LINK_LIBRARIES_KEYWORD is not available
  # So this is a workaround to append PRIVATE in front of it
  set(CUDA_LIBRARIES ${CUDA_LINK_LIBRARIES_KEYWORD} ${CUDA_LIBRARIES})
endif()
cuda_add_library(${dali_operator_lib} STATIC ${DALI_OPERATOR_SRCS})
target_link_libraries(${dali_operator_lib} PRIVATE "${dali_kernel_lib}")

cuda_add_library(${dali_lib} SHARED ${DALI_SRCS} ${DALI_PROTO_OBJ})

# Options for using Dockerfile FFmpeg version
target_compile_definitions(${dali_lib} PRIVATE HAVE_AVSTREAM_CODECPAR=1)
target_compile_definitions(${dali_lib} PRIVATE HAVE_AVBSFCONTEXT=1)

target_compile_definitions(${dali_operator_lib} PRIVATE HAVE_AVSTREAM_CODECPAR=1)
target_compile_definitions(${dali_operator_lib} PRIVATE HAVE_AVBSFCONTEXT=1)

# Define symbol version script for libdali.so
set(exports_map_path "${PROJECT_BINARY_DIR}/stage")
set(dali_lib_exports "lib${dali_lib}.map")
configure_file("${PROJECT_SOURCE_DIR}/cmake/${dali_lib_exports}.in" "${exports_map_path}/${dali_lib_exports}")
target_link_libraries(${dali_lib} PRIVATE ${dali_operator_lib})
# link whole content of dali_operator_lib into dali_lib
target_link_libraries(${dali_lib} PRIVATE "-Wl,--whole-archive $<TARGET_LINKER_FILE:${dali_operator_lib}> -Wl,--no-whole-archive")

target_link_libraries(${dali_lib} PRIVATE -Wl,--version-script=${exports_map_path}/${dali_lib_exports})
add_custom_target(${dali_lib_exports} DEPENDS "${exports_map_path}/${dali_lib_exports}")
add_dependencies(${dali_lib} ${dali_lib_exports})

# Exclude (most) statically linked dali dependencies from the exports of libdali.so
string(REPLACE ";" ":" exclude_libs "${DALI_EXCLUDES}")
message(STATUS "Exclude libs '${exclude_libs}'")
target_link_libraries(${dali_lib} PRIVATE "-Wl,--exclude-libs,${exclude_libs}")

# Link in dali's dependencies
message(STATUS "Adding dependencies to ${dali_lib}: '${DALI_LIBS}'")
target_link_libraries(${dali_lib} PRIVATE ${DALI_LIBS})

# install libdali to the wheel dir
set_target_properties(${dali_lib} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/
)

set_target_properties(${dali_operator_lib} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# filter all *.h which don't have 'test' in name, create pairs 'src dst' string with sed and call xargs
# with install command to copy with dest recursive dir creation
add_custom_target(install_headers ALL
    DEPENDS ${dali_lib}
)

add_custom_command(
    TARGET install_headers
    COMMAND find ${CMAKE_CURRENT_LIST_DIR} -regex ".*\\\\.h" | grep --invert test
    | sed -r 's,\(^${CMAKE_CURRENT_LIST_DIR}\)\(.*\),\\1\\2 ${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/include/dali\\2,'
    | xargs -n 2 install -D
  )

################################################
# Build test suite
################################################
if (BUILD_TEST)
  set(TEST_BINARY_DIR ${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/test)

  # get all test srcs
  add_subdirectory(test)

  set(test_main_bin "dali_test.bin")
  cuda_add_executable(${test_main_bin} "${DALI_TEST_SRCS}")

  # Link to the dali lib
  message(STATUS "Adding dependencies to ${test_main_bin}: '${dali_lib}'")
  add_dependencies(${test_main_bin} ${dali_lib} ${dali_kernel_test_lib} ${dali_kernel_lib})

  # We'll have to add dependency libs
  target_link_libraries(${test_main_bin} PRIVATE
    "-Wl,--whole-archive $<TARGET_LINKER_FILE:${dali_kernel_test_lib}> -Wl,--no-whole-archive"
    "${dali_kernel_test_lib}"
    "${DALI_LIBS}"
    "${dali_lib}"
    "gtest")

  set_target_properties(${test_main_bin} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})
endif()


################################################
# Build benchmark suite
################################################
if (BUILD_BENCHMARK)
  # get benchmark main
  add_subdirectory(benchmark)

  set(benchmark_bin "dali_benchmark.bin")
  cuda_add_executable(${benchmark_bin} "${DALI_BENCHMARK_SRCS}")

  # Link to the dali lib
  message(STATUS "Adding dependencies to ${benchmark_bin}: '${dali_lib}'")
  add_dependencies(${benchmark_bin} ${dali_lib})

  target_link_libraries(${benchmark_bin} PRIVATE ${DALI_LIBS} ${dali_lib} benchmark pthread)

  set_target_properties(${benchmark_bin} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/test)
endif()


################################################
# Build the TensorFlow op
################################################
if (BUILD_TENSORFLOW)

  add_subdirectory(tensorflow)

  # get default TF installation path
   execute_process(
     COMMAND python -c "from __future__ import print_function; import sys; paths = sys.path; f = [p for p in paths if \"-packages\" in p] ; print(f[0], end=\"\")"
     OUTPUT_VARIABLE DIST_PACKAGES_DIR)

  set(TF_PATHS)
  set(TF_VER)
  # obtain custom versions of TF we want to build aganist
  file(GLOB TF_VERSIONS RELATIVE /tensorflow /tensorflow/*)
  foreach(VER ${TF_VERSIONS})
    set(FULL_PATH /tensorflow/${VER})
    if(IS_DIRECTORY ${FULL_PATH})
      list(APPEND TF_PATHS ${FULL_PATH})
      list(APPEND TF_VER "_${VER}")
    ENDIF()
  ENDFOREACH()

  # append default TF version
  list(APPEND TF_PATHS ${DIST_PACKAGES_DIR})
  list(APPEND TF_VER ";")

  # build TF plugin for each TF version we want to support
  list(LENGTH TF_PATHS TF_LIST_L)
  math(EXPR TF_LIST_L_ITER "${TF_LIST_L} - 1")
  foreach(IDX RANGE ${TF_LIST_L_ITER})
    list(GET TF_PATHS ${IDX} FULL_PATH)
    list(GET TF_VER ${IDX} VER)
    set(customop_lib "dali_tf${VER}")

    cuda_add_library(${customop_lib} SHARED ${DALI_TF_SRCS})

    separate_arguments(TF_CFLAGS UNIX_COMMAND "-I${FULL_PATH}/tensorflow/include -I${FULL_PATH}/external/nsync/public -D_GLIBCXX_USE_CXX11_ABI=0")
    target_compile_options(${customop_lib} PRIVATE ${TF_CFLAGS})

    message(STATUS "Adding dependencies to ${customop_lib}: '${dali_lib}'")
    add_dependencies(${customop_lib} ${dali_lib})

    target_link_libraries(${customop_lib} PRIVATE -L${FULL_PATH}/tensorflow -ltensorflow_framework)
    target_link_libraries(${customop_lib} PRIVATE ${dali_lib})

    set_target_properties(${customop_lib} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/plugin")
  ENDFOREACH()
endif()


################################################
# Build the DALI python bindings
################################################
if (BUILD_PYTHON)
  # Get all python srcs
  add_subdirectory(python)
endif()
