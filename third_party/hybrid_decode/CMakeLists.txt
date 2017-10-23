# NOTE(tgale): Haven't tried with anything else.
cmake_minimum_required(VERSION 3.5)

project(HybridDecode CXX)

# Helper function to remove elements from a variable
function (remove TARGET INPUT)
  foreach(ITEM ${ARGN})
    list(REMOVE_ITEM INPUT "${ITEM}")
  endforeach()
  set(${TARGET} ${INPUT} PARENT_SCOPE)
endfunction(remove)

# Default to release build
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type from [Debug, Release]. For perf testing, build Release" FORCE)
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message(STATUS "Building in DEBUG mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")
endif()

# Cmake path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Dependencies
include(cmake/Dependencies.cmake)

# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

# TODO(tgale): Add sm targets
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

# Project dir
include_directories(${PROJECT_SOURCE_DIR}/include)
cuda_include_directories(${PROJECT_SOURCE_DIR}/include)

# Project build
add_subdirectory(src)

# HACK: Install rule for headers
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR} DESTINATION include/third_party
  FILES_MATCHING PATTERN "*.h" PATTERN "test*.h" EXCLUDE)
