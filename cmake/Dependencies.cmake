# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
# For CUDA
find_package(CUDA REQUIRED)
if (${CUDA_VERSION_MAJOR} LESS 8)
  message(FATAL "DALI needs at least CUDA 8.0")
endif()
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${CUDA_LIBRARIES})
list(APPEND DALI_EXCLUDES libcudart_static.a)

# For NVJPEG
if (BUILD_NVJPEG)
  find_package(NVJPEG REQUIRED)
  include_directories(SYSTEM ${NVJPEG_INCLUDE_DIRS})
  list(APPEND DALI_LIBS ${NVJPEG_LIBRARY})
  list(APPEND DALI_EXCLUDES libnvjpeg_static.a)
  add_definitions(-DDALI_USE_NVJPEG)
endif()

# For NPP
find_cuda_helper_libs(nppc_static)
if (${CUDA_VERSION_MAJOR} EQUAL 8)
# In CUDA 8 Nppi is a single library
find_cuda_helper_libs(nppi_static)
list(APPEND DALI_LIBS ${CUDA_nppc_static_LIBRARY}
                      ${CUDA_nppi_static_LIBRARY})
list(APPEND DALI_EXCLUDES libnppc_static.a
                          libnppi_static.a)
else()
find_cuda_helper_libs(nppicom_static)
find_cuda_helper_libs(nppicc_static)
find_cuda_helper_libs(nppig_static)
list(APPEND DALI_LIBS ${CUDA_nppc_static_LIBRARY}
                      ${CUDA_nppicom_static_LIBRARY}
                      ${CUDA_nppicc_static_LIBRARY}
                      ${CUDA_nppig_static_LIBRARY})
list(APPEND DALI_EXCLUDES libnppc_static.a
                          libnppicom_static.a
                          libnppicc_static.a
                          libnppig_static.a)
endif()

# CULIBOS needed when using static CUDA libs
find_cuda_helper_libs(culibos)
list(APPEND DALI_LIBS ${CUDA_culibos_LIBRARY})
list(APPEND DALI_EXCLUDES libculibos.a)

# NVTX for profiling
if (BUILD_NVTX)
  find_cuda_helper_libs(nvToolsExt)
  list(APPEND DALI_LIBS ${CUDA_nvToolsExt_LIBRARY})
  add_definitions(-DDALI_USE_NVTX)
endif()

# Google C++ testing framework
if (BUILD_TEST)
  set(BUILD_GTEST ON CACHE INTERNAL "Builds gtest submodule")
  set(BUILD_GMOCK OFF CACHE INTERNAL "Builds gmock submodule")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
endif()

# Google Benchmark
if (BUILD_BENCHMARK)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "Build benchmark testsuite")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/benchmark EXCLUDE_FROM_ALL)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/benchmark/include/benchmark)
endif()

# LibJpegTurbo
if (BUILD_JPEG_TURBO)
  find_package(JpegTurbo REQUIRED)
  include_directories(SYSTEM ${JPEG_TURBO_INCLUDE_DIR})
  list(APPEND DALI_LIBS ${JPEG_TURBO_LIBRARY})
  list(APPEND DALI_EXCLUDES libturbojpeg.a)
  add_definitions(-DDALI_USE_JPEG_TURBO)
endif()

# OpenCV
# Note: OpenCV 3.* 'imdecode()' is in the imgcodecs library
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
if (OpenCV_FOUND)
  string(SUBSTRING ${OpenCV_VERSION} 0 1 OCV_VERSION)
  if ("${OCV_VERSION}" STREQUAL "3")
    # Get the imgcodecs library
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  else()
    # For opencv 2.x, image encode/decode functions are in highgui
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
  endif()

  # Check for opencv
  message(STATUS "Found OpenCV ${OpenCV_VERSION} (libs: ${OpenCV_LIBRARIES})")
endif()
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${OpenCV_LIBRARIES})
list(APPEND DALI_EXCLUDES libopencv_core.a;libopencv_imgproc.a;libopencv_imgcodecs.a)

# PyBind
if (BUILD_PYTHON)
  # Build w/ c++11
  set(PYBIND11_CPP_STANDARD -std=c++11)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11)
endif()

# LMDB
if (BUILD_LMDB)
  find_package(LMDB)
  if (LMDB_FOUND)
    message(STATUS "Found LMDB ${LMDB_INCLUDE_DIR} : ${LMDB_LIBRARIES}")
    include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
    list(APPEND DALI_LIBS ${LMDB_LIBRARIES})
    list(APPEND DALI_EXCLUDES liblmdb.a)
  else()
    message(STATUS "LMDB not found")
  endif()
endif()

# protobuf
find_package(Protobuf REQUIRED)

if (PROTOBUF_FOUND)
  # Determine if we have proto v2.x or 3.x
  execute_process(COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --version COMMAND cut -d " " -f 2 COMMAND cut -d . -f 1
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PROTOBUF_VERSION RESULT_VARIABLE __res)
  message(STATUS "Found Protobuf version ${PROTOBUF_VERSION} : ${PROTOBUF_INCLUDE_DIRS} : ${PROTOBUF_LIBRARY}")

  if (PROTOBUF_VERSION MATCHES 3)
    add_definitions(-DDALI_BUILD_PROTO3=1)
    set(BUILD_PROTO3 ON CACHE STRING "Build proto3")
  endif()
  include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIRS})
  list(APPEND DALI_LIBS ${PROTOBUF_LIBRARY})
  ################
  ### Don't exclude protobuf symbols; doing so will lead to segfaults
  #list(APPEND DALI_EXCLUDES libprotobuf.a)
  ################
else()
  message(STATUS "Protobuf not found")
endif()
