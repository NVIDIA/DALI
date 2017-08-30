# NOTE(tgale): Haven't tried with anything else.
cmake_minimum_required(VERSION 3.5)

project(NDLL CXX)

# Build options
option(BUILD_TEST "Build googletest test suite" ON)

# Dependencies
include(cmake/Dependencies.cmake)

# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# TODO(tgale): Add sm targets
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

# Project dir
include_directories(${PROJECT_SOURCE_DIR})
cuda_include_directories(${PROJECT_SOURCE_DIR})

# Project build
add_subdirectory(ndll)


