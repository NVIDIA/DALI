# NOTE(tgale): Haven't tried with anything else.
cmake_minimum_required(VERSION 3.5)

project(NDLL CXX)

# Build options
option(BUILD_TEST "Build googletest test suite" ON)
option(BUILD_BENCHMARK "Build benchmark suite" ON)
option(USE_NVTX "Build with NVTX profiling enabled" ON)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-unused-function -fno-strict-aliasing -O2 -fPIC")

# TODO(tgale): Add sm targets
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")

# Project dir
include_directories(${PROJECT_SOURCE_DIR})
cuda_include_directories(${PROJECT_SOURCE_DIR})

# Project build
add_subdirectory(ndll)


