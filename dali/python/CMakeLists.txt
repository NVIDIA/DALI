# Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if (PREBUILD_DALI_LIBS)
# find prebuild DALI libs
  find_library(PREBUILD_DALI_LIB NAMES dali)
  find_library(PREBUILD_DALI_OPERATORS_LIB NAMES dali_operators)
  find_library(PREBUILD_DALI_KERNELES_LIB NAMES dali_kernels)
  find_library(PREBUILD_DALI_CORE_LIB NAMES dali_core)
  build_per_python_lib(dali_python
                       OUTPUT_NAME backend_impl
                       OUTPUT_DIR ${DALI_LIBRARY_OUTPUT_DIR}
                       PUBLIC_LIBS ${PREBUILD_DALI_LIB} ${PREBUILD_DALI_OPERATORS_LIB} ${PREBUILD_DALI_KERNELES_LIB} ${PREBUILD_DALI_CORE_LIB} ${CUDART_LIB}
                       PRIV_LIBS ${CUDA_LIBRARIES} dynlink_cuda
                       EXCLUDE_LIBS ${exclude_libs}
                       SRC ${DALI_PYTHON_BACKEND_SRCS})
else (PREBUILD_DALI_LIBS)
  build_per_python_lib(dali_python
                       OUTPUT_NAME backend_impl
                       OUTPUT_DIR ${DALI_LIBRARY_OUTPUT_DIR}
                       PUBLIC_LIBS dali dali_operators dali_kernels dali_core ${CUDART_LIB}
                       PRIV_LIBS ${CUDA_LIBRARIES} dynlink_cuda
                       EXCLUDE_LIBS ${exclude_libs}
                       SRC ${DALI_PYTHON_BACKEND_SRCS})
endif (PREBUILD_DALI_LIBS)

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
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/tools/wds2idx.py" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/Acknowledgements.txt" "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/COPYRIGHT" "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/internal_tools/find_dali.cmake" "${PROJECT_BINARY_DIR}/dali/cmake")

if(NOT ${CMAKE_CROSSCOMPILING} AND NOT ${BUILD_WITH_ASAN})
  # Prepate PYTHONPATH targetting the pre-packed wheel produced in this build
  set(PYTHON_TARGET_PATH ${PROJECT_BINARY_DIR}/dali/python)
  if($ENV{PYTHONPATH})
    set(PYTHONPATH "${PYTHON_TARGET_PATH}:$ENV{PYTHONPATH}")
  else()
    set(PYTHONPATH "${PYTHON_TARGET_PATH}")
  endif()

  if (PREBUILD_DALI_LIBS)
    add_custom_target(dali_python_generate_stubs ALL
                      DEPENDS dali_python ${dali_python_function_lib})
  else (PREBUILD_DALI_LIBS)
    add_custom_target(dali_python_generate_stubs ALL
                      DEPENDS dali_python ${dali_python_function_lib} dali dali_operators copy_post_build_target)
  endif (PREBUILD_DALI_LIBS)

  # Build the .pyi stubs, adjusting the PYTHONPATH for the invocation, allowing to use the
  # backend from the current build.
  add_custom_command(
    TARGET dali_python_generate_stubs
    POST_BUILD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND PYTHONPATH=${PYTHONPATH} ${PYTHON_STUBGEN_INTERPRETER} ${PROJECT_SOURCE_DIR}/internal_tools/python_stub_generator.py --wheel_path ${PROJECT_BINARY_DIR}/dali/python/nvidia/dali
  )

  # Format the bindings, so they are somewhat readable.
  if (${FORMAT_PYTHON_STUB_FILES})
    add_custom_target(dali_python_format_stubs ALL DEPENDS dali_python dali_python_generate_stubs)
    add_custom_command(
      TARGET dali_python_format_stubs
      POST_BUILD
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMAND find dali/python/nvidia/dali -name '*.pyi' -print0 | xargs -0 black --line-length 100 --target-version py38 --verbose
    )
  endif()
endif()
