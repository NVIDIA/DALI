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

set(DALI_TOP_DIR "${PROJECT_SOURCE_DIR}/..")

include(${DALI_TOP_DIR}/cmake/Utils.cmake)
get_dali_version(${DALI_TOP_DIR}/VERSION DALI_VERSION)
if (DALI_BUILD_FLAVOR)
  set(DALI_FLAVOR "${DALI_BUILD_FLAVOR} ")
  set(DALI_FLAVOR_MINUS "-${DALI_BUILD_FLAVOR}")
  set(DALI_VERSION "${DALI_VERSION}.${TIMESTAMP}")
endif()

configure_file("${PROJECT_SOURCE_DIR}/setup.py.in" "${PROJECT_BINARY_DIR}/setup.py")
file(GLOB PREBUILT_PLUGINS "libdali_tf_*.so")
file(COPY ${PREBUILT_PLUGINS} DESTINATION "${PROJECT_BINARY_DIR}")
file(COPY "daliop.cc" DESTINATION "${PROJECT_BINARY_DIR}")
file(COPY "${PROJECT_SOURCE_DIR}/MANIFEST.in" DESTINATION "${PROJECT_BINARY_DIR}")
file(COPY "${DALI_TOP_DIR}/Acknowledgements.txt" DESTINATION "${PROJECT_BINARY_DIR}")
file(COPY "${DALI_TOP_DIR}/COPYRIGHT" DESTINATION "${PROJECT_BINARY_DIR}")
file(COPY "${DALI_TOP_DIR}/LICENSE" DESTINATION "${PROJECT_BINARY_DIR}")
