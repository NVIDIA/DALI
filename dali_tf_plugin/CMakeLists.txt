# Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.25.2)
project(dali_tf)

set(DALI_ROOT "${PROJECT_SOURCE_DIR}/..")

SET(CUDA_VERSION "" CACHE STRING "CUDA version")
if ("${CUDA_VERSION}" STREQUAL "")
  message(FATAL_ERROR "CUDA version not provided. Pass CUDA version to cmake (e.g -DCUDA_VERSION:STRING=\"10.1\")")
endif()

include(${DALI_ROOT}/cmake/Utils.cmake)
parse_cuda_version(${CUDA_VERSION} CUDA_VERSION_MAJOR CUDA_VERSION_MINOR CUDA_VERSION_PATCH CUDA_VERSION_SHORT CUDA_VERSION_SHORT_DIGIT_ONLY)

get_dali_version(${DALI_ROOT}/VERSION DALI_VERSION)
if (DALI_BUILD_FLAVOR)
  set(DALI_FLAVOR "${DALI_BUILD_FLAVOR} ")
  set(DALI_FLAVOR_MINUS "-${DALI_BUILD_FLAVOR}")
  set(DALI_VERSION "${DALI_VERSION}.${TIMESTAMP}")
endif()

configure_file("${PROJECT_SOURCE_DIR}/setup.py.in" "${PROJECT_BINARY_DIR}/setup.py")
configure_file("${PROJECT_SOURCE_DIR}/nvidia/dali_tf_plugin/__init__.py.in" "${PROJECT_BINARY_DIR}/nvidia/dali_tf_plugin/__init__.py")
install(FILES "${PROJECT_SOURCE_DIR}/nvidia/dali_tf_plugin/dali_tf_plugin.py" DESTINATION "${PROJECT_BINARY_DIR}/nvidia/dali_tf_plugin/")

install(DIRECTORY "${DALI_ROOT}/include" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "${DALI_ROOT}/tools/stubgen.py" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "dali_dataset.h" DESTINATION "${PROJECT_BINARY_DIR}/dali_tf_plugin")
install(FILES "dali_helper.h" DESTINATION "${PROJECT_BINARY_DIR}/dali_tf_plugin")
install(FILES "daliop.cc" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "dali_dataset_op.cc" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "dali_tf_plugin_install_tool.py" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "dali_tf_plugin_utils.py" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "${PROJECT_SOURCE_DIR}/MANIFEST.in" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "${DALI_ROOT}/Acknowledgements.txt" DESTINATION "${PROJECT_BINARY_DIR}/nvidia/dali_tf_plugin")
install(FILES "${DALI_ROOT}/COPYRIGHT" DESTINATION "${PROJECT_BINARY_DIR}/nvidia/dali_tf_plugin")
install(FILES "${DALI_ROOT}/LICENSE" DESTINATION "${PROJECT_BINARY_DIR}/nvidia/dali_tf_plugin")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/prebuilt" DESTINATION "${PROJECT_BINARY_DIR}" OPTIONAL)
