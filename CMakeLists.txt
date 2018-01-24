# NOTE(tgale): Haven't tried with anything else.
cmake_minimum_required(VERSION 3.5)

project(NDLL CXX)

# Build options
option(BUILD_TEST "Build googletest test suite" ON)
option(BUILD_BENCHMARK "Build benchmark suite" ON)
option(USE_NVTX "Build with NVTX profiling enabled" ON)
option(BUILD_PYTHON "Build python bindings" ON)
option(BUILD_LMDB "Build LMDB readers" OFF)

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -DNDLL_DEBUG=1")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G -DNDLL_DEBUG=1")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDLL_DEBUG=0")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DNDLL_DEBUG=0")
endif()

# Cmake path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Dependencies
include(cmake/Dependencies.cmake)

# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-unused-variable -Wno-unused-function -fno-strict-aliasing -O2 -fPIC")

# Add ptx & bin flags for cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \
  -gencode arch=compute_30,code=sm_30 \
  -gencode arch=compute_35,code=sm_35 \
  -gencode arch=compute_50,code=sm_50 \
  -gencode arch=compute_52,code=sm_52 \
  -gencode arch=compute_60,code=sm_60 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_70,code=compute_70")

# Project dir
include_directories(BEFORE ${PROJECT_SOURCE_DIR})
include_directories(BEFORE ${PROJECT_BINARY_DIR})
cuda_include_directories(${PROJECT_SOURCE_DIR})

# Project build
add_subdirectory(ndll)

# HACK: Add __init__.pys as needed
file(WRITE ${CMAKE_BINARY_DIR}/ndll/__init__.py "")

add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/lint.cmake)

