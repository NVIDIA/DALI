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

# Get all the srcs
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_PYTHON_BACKEND_SRCS)
pybind11_add_module(${dali_python_lib} MODULE ${DALI_PYTHON_BACKEND_SRCS})

target_link_libraries(${dali_python_lib} PUBLIC
  ${dali_lib} ${dali_operator_lib} ${dali_kernel_lib} ${dali_core_lib})
target_link_libraries(${dali_python_lib} PRIVATE
  ${CUDA_LIBRARIES})

# Set output directory for backend_impl
set_target_properties(${dali_python_lib} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${DALI_LIBRARY_OUTPUT_DIR})

if (DALI_BUILD_FLAVOR)
  set(DALI_FLAVOR "${DALI_BUILD_FLAVOR} ")
  set(DALI_FLAVOR_MINUS "-${DALI_BUILD_FLAVOR}")
  set(DALI_VERSION "${DALI_VERSION}.${TIMESTAMP}")
endif()

# Add the COPYRIGHT, LICENSE, and Acknowledgements
copy_post_build(${dali_python_lib} "${PROJECT_SOURCE_DIR}/dali/python/nvidia" "${PROJECT_BINARY_DIR}/dali/python")
configure_file("${PROJECT_SOURCE_DIR}/dali/python/__init__.py.in" "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/__init__.py")
configure_file("${PROJECT_SOURCE_DIR}/dali/python/setup.py.in" "${PROJECT_BINARY_DIR}/stage/setup.py")
copy_post_build(${dali_python_lib} "${PROJECT_BINARY_DIR}/stage/setup.py" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(${dali_python_lib} "${PROJECT_SOURCE_DIR}/dali/python/MANIFEST.in" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(${dali_python_lib} "${PROJECT_SOURCE_DIR}/tools/rec2idx.py" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(${dali_python_lib} "${PROJECT_SOURCE_DIR}/tools/tfrecord2idx" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(${dali_python_lib} "${PROJECT_SOURCE_DIR}/Acknowledgements.txt" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(${dali_python_lib} "${PROJECT_SOURCE_DIR}/COPYRIGHT" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(${dali_python_lib} "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_BINARY_DIR}/dali/python")
