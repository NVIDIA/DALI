# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

##################################################################
# Google C++ testing framework
##################################################################
if (BUILD_TEST)
  set(BUILD_GTEST ON CACHE INTERNAL "Build gtest submodule")
  set(BUILD_GMOCK OFF CACHE INTERNAL "Build gmock submodule")
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
endif()

##################################################################
# Google Benchmark
##################################################################
if (BUILD_BENCHMARK)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "Build benchmark testsuite")
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/third_party/benchmark EXCLUDE_FROM_ALL)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/benchmark/include/benchmark)
endif()

##################################################################
# CUDA Toolkit libraries (including NVJPEG)
##################################################################
# Note: CUDA 8 support is unofficial.  CUDA 9 is officially supported
find_package(CUDA 8.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${CUDA_LIBRARIES})
list(APPEND DALI_EXCLUDES libcudart_static.a)

# For NVJPEG
if (BUILD_NVJPEG)
  find_package(NVJPEG 9.0 REQUIRED)
  if(${CUDA_VERSION} VERSION_LESS ${NVJPEG_VERSION})
    message(WARNING "Using nvJPEG ${NVJPEG_VERSION} together with CUDA ${CUDA_VERSION} "
                    "requires NVIDIA drivers compatible with CUDA ${NVJPEG_VERSION} or later")
  endif()
  include_directories(SYSTEM ${NVJPEG_INCLUDE_DIR})
  list(APPEND DALI_LIBS ${NVJPEG_LIBRARY})
  list(APPEND DALI_EXCLUDES libnvjpeg_static.a)
  add_definitions(-DDALI_USE_NVJPEG)
else()
  # Note: Support for disabling nvJPEG is unofficial
  message(STATUS "Building WITHOUT nvJPEG")
endif()

# NVIDIA NPPC library
find_cuda_helper_libs(nppc_static)
list(APPEND DALI_LIBS ${CUDA_nppc_static_LIBRARY})
list(APPEND DALI_EXCLUDES libnppc_static.a)

# NVIDIA NPPI library
if (${CUDA_VERSION} VERSION_LESS "9.0")
  # In CUDA 8, NPPI is a single library
  find_cuda_helper_libs(nppi_static)
  list(APPEND DALI_LIBS ${CUDA_nppi_static_LIBRARY})
  list(APPEND DALI_EXCLUDES libnppi_static.a)
else()
  find_cuda_helper_libs(nppicom_static)
  find_cuda_helper_libs(nppicc_static)
  find_cuda_helper_libs(nppig_static)
  list(APPEND DALI_LIBS ${CUDA_nppicom_static_LIBRARY}
                        ${CUDA_nppicc_static_LIBRARY}
                        ${CUDA_nppig_static_LIBRARY})
  list(APPEND DALI_EXCLUDES libnppicom_static.a
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

##################################################################
# libjpeg-turbo
##################################################################
if (BUILD_JPEG_TURBO)
  find_package(JpegTurbo 1.5 REQUIRED)
  include_directories(SYSTEM ${JpegTurbo_INCLUDE_DIR})
  list(APPEND DALI_LIBS ${JpegTurbo_LIBRARY})
  list(APPEND DALI_EXCLUDES libturbojpeg.a)
  add_definitions(-DDALI_USE_JPEG_TURBO)
else()
  # Note: Support for disabling libjpeg-turbo is unofficial
  message(STATUS "Building WITHOUT JpegTurbo")
endif()

##################################################################
# OpenCV
##################################################################
# For OpenCV 3.x, 'imdecode()' is in the imgcodecs library
find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc imgcodecs)
if(NOT OpenCV_FOUND)
  # Note: OpenCV 2 support is unofficial
  # For OpenCV 2.x, image encode/decode functions are in highgui
  find_package(OpenCV 2.0 REQUIRED COMPONENTS core imgproc highgui)
endif()
message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} (found suitable version \"${OpenCV_VERSION}\", minimum required is \"2.0\")")
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${OpenCV_LIBRARIES})
list(APPEND DALI_EXCLUDES libopencv_core.a;libopencv_imgproc.a;libopencv_highgui.a;libopencv_imgcodecs.a)

##################################################################
# PyBind
##################################################################
if (BUILD_PYTHON)
  set(PYBIND11_CPP_STANDARD -std=c++11)
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/third_party/pybind11)
endif()

##################################################################
# LMDB
##################################################################
if (BUILD_LMDB)
  find_package(LMDB 0.9 REQUIRED)
  include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
  list(APPEND DALI_LIBS ${LMDB_LIBRARIES})
  list(APPEND DALI_EXCLUDES liblmdb.a)
endif()

##################################################################
# protobuf
##################################################################
find_package(Protobuf 2.0 REQUIRED)
if(${Protobuf_VERSION} VERSION_LESS "3.0")
  message(STATUS "TensorFlow TFRecord file format support is not available with Protobuf 2")
else()
  message(STATUS "Enabling TensorFlow TFRecord file format support")
  add_definitions(-DDALI_BUILD_PROTO3=1)
  set(BUILD_PROTO3 ON CACHE STRING "Build proto3")
endif()
include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${PROTOBUF_LIBRARY})
## Don't exclude protobuf symbols here; doing so will lead to segfaults
## Instead we'll use EXPORT_MACRO DLL_PUBLIC later in dali/*/CMakeLists.txt to
## tell protobuf how to hide things that don't specifically need to be exported
#list(APPEND DALI_EXCLUDES libprotobuf.a)
