# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

cmake_minimum_required(VERSION 3.2)

set(DALI_ROOT "${PROJECT_SOURCE_DIR}/..")

include(${DALI_ROOT}/cmake/Utils.cmake)
get_dali_version(${DALI_ROOT}/VERSION DALI_VERSION)
if (DALI_BUILD_FLAVOR)
  set(DALI_FLAVOR "${DALI_BUILD_FLAVOR} ")
  set(DALI_FLAVOR_MINUS "-${DALI_BUILD_FLAVOR}")
  set(DALI_VERSION "${DALI_VERSION}.${TIMESTAMP}")
endif()

# Note: CUDA 8 support is unofficial. CUDA 9 is officially supported
find_package(CUDA 8.0 REQUIRED)

configure_file("${PROJECT_SOURCE_DIR}/setup.py.in" "${PROJECT_BINARY_DIR}/setup.py")
install(FILES "daliop.cc" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "dali_dataset_op.cc" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "dali_tf_plugin_install_tool.py" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "dali_tf_plugin_utils.py" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "${PROJECT_SOURCE_DIR}/MANIFEST.in" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "${DALI_ROOT}/Acknowledgements.txt" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "${DALI_ROOT}/COPYRIGHT" DESTINATION "${PROJECT_BINARY_DIR}")
install(FILES "${DALI_ROOT}/LICENSE" DESTINATION "${PROJECT_BINARY_DIR}")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/prebuilt" DESTINATION "${PROJECT_BINARY_DIR}")
