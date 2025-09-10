# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.25.2)

project(dali_plugins)

if(NOT DEFINED PYTHON_EXECUTABLE)
  find_package(Python COMPONENTS Interpreter)
  set(PYTHONINTERP_FOUND ${Python_Interpreter_FOUND})
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
endif()

if(NOT DEFINED PYTHON_EXECUTABLE)
  message(FATAL_ERROR "No Python executable found.")
endif()
message(STATUS "Using Python ${PYTHON_EXECUTABLE}")

set(DALI_ROOT "${PROJECT_SOURCE_DIR}/..")
include(${DALI_ROOT}/cmake/Utils.cmake)
get_dali_version(${DALI_ROOT}/VERSION DALI_VERSION)

if(DALI_BUILD_FLAVOR)
  set(DALI_FLAVOR "${DALI_BUILD_FLAVOR} ")
  set(DALI_FLAVOR_MINUS "-${DALI_BUILD_FLAVOR}")
  set(DALI_VERSION "${DALI_VERSION}")
endif()

add_subdirectory(video)