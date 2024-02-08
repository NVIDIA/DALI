# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18)

project(dali_video)

set(DALI_ROOT "${PROJECT_SOURCE_DIR}/../..")
SET(CUDA_VERSION "" CACHE STRING "CUDA version")
if("${CUDA_VERSION}" STREQUAL "")
  message(FATAL_ERROR "CUDA version not provided. Pass CUDA version to cmake (e.g -DCUDA_VERSION:STRING=\"10.1\")")
endif()

include(${DALI_ROOT}/cmake/Utils.cmake)
parse_cuda_version(${CUDA_VERSION} CUDA_VERSION_MAJOR CUDA_VERSION_MINOR CUDA_VERSION_PATCH CUDA_VERSION_SHORT CUDA_VERSION_SHORT_DIGIT_ONLY)

get_dali_version(${DALI_ROOT}/VERSION DALI_VERSION)

if(DALI_BUILD_FLAVOR)
  set(DALI_FLAVOR "${DALI_BUILD_FLAVOR} ")
  set(DALI_FLAVOR_MINUS "-${DALI_BUILD_FLAVOR}")
  set(DALI_VERSION "${DALI_VERSION}.${TIMESTAMP}")
endif()

# This project creates the project to pack as a source distribution
file(COPY "pkg_src" DESTINATION ".") # copy to build folder
file(COPY "src" DESTINATION "pkg_src") # copy to build folder
file(COPY "${DALI_ROOT}/Acknowledgements.txt" DESTINATION "pkg_src")
file(COPY "${DALI_ROOT}/COPYRIGHT" DESTINATION "pkg_src")
file(COPY "${DALI_ROOT}/LICENSE" DESTINATION "pkg_src")
configure_file("setup.py.in" "pkg_src/setup.py")
configure_file("src/nvidia/dali_video/__init__.py.in" "pkg_src/src/nvidia/dali_video/__init__.py")

if(NOT DEFINED PYTHON_EXECUTABLE)
  find_package(Python COMPONENTS Interpreter)
  set(PYTHONINTERP_FOUND ${Python_Interpreter_FOUND})
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
endif()

if(NOT DEFINED PYTHON_EXECUTABLE)
  message(FATAL_ERROR "No Python executable found.")
endif()
message(STATUS "Using Python ${PYTHON_EXECUTABLE}")

add_custom_target(dali-video-sdist)
add_custom_command(
    TARGET dali-video-sdist
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pkg_src
    COMMAND ${PYTHON_EXECUTABLE} setup.py sdist)
