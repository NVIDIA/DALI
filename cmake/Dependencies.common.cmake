# Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


##################################################################
# OpenCV
##################################################################
if (BUILD_OPENCV)
  # For OpenCV 3 and later, 'imdecode()' is in the imgcodecs library

  find_package(OpenCV 4.0 QUIET COMPONENTS core imgproc imgcodecs)
  if(NOT OpenCV_FOUND)
    find_package(OpenCV 3.0 REQUIRED COMPONENTS core imgproc imgcodecs)
  endif()

  message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} (found suitable version \"${OpenCV_VERSION}\", minimum required is \"3.0\")")
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND DALI_LIBS ${OpenCV_LIBRARIES})
  message("OpenCV libraries: ${OpenCV_LIBRARIES}")
  list(APPEND DALI_EXCLUDES libopencv_core.a;libopencv_imgproc.a;libopencv_highgui.a;libopencv_imgcodecs.a;liblibwebp.a;libittnotify.a;libpng.a;liblibtiff.a;liblibjasper.a;libIlmImf.a;liblibjpeg-turbo.a)
endif()

##################################################################
#
# Optional dependencies
#
##################################################################

##################################################################
# Google C++ testing framework
##################################################################
if (BUILD_TEST)
  set(BUILD_GTEST ON CACHE INTERNAL "Build gtest submodule")
  set(BUILD_GMOCK OFF CACHE INTERNAL "Build gmock submodule")
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
  set_target_properties(gtest PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

##################################################################
# Google Benchmark
##################################################################
if (BUILD_BENCHMARK)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "Build benchmark testsuite")
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/third_party/benchmark EXCLUDE_FROM_ALL)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/benchmark/include/benchmark)
  set_target_properties(benchmark PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

##################################################################
# libjpeg-turbo
##################################################################
if (BUILD_JPEG_TURBO)
  find_package(JPEG 62 REQUIRED) # 1.5.3 version
  include_directories(${JPEG_INCLUDE_DIR})
  message("Using libjpeg-turbo at ${JPEG_LIBRARY}")
  list(APPEND DALI_LIBS ${JPEG_LIBRARY})
  add_definitions(-DDALI_USE_JPEG_TURBO)
endif()

##################################################################
# libtiff
##################################################################
if (BUILD_LIBTIFF)
  find_package(TIFF REQUIRED)
  include_directories(${TIFF_INCLUDE_DIR})
  message("Using libtiff at ${TIFF_LIBRARY}")
  list(APPEND DALI_LIBS ${TIFF_LIBRARY})
endif()

##################################################################
# PyBind
##################################################################
if (BUILD_PYTHON)
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
# libsnd
##################################################################
if(BUILD_LIBSND)
  find_library(libsnd_LIBS
          NAMES sndfile libsndfile
          PATHS ${LIBSND_ROOT_DIR} "/usr/local" ${CMAKE_SYSTEM_PREFIX_PATH}
          PATH_SUFFIXES lib lib64)
  if(${libsnd_LIBS} STREQUAL libsnd_LIBS-NOTFOUND)
    message(FATAL_ERROR "libsnd (sndfile) could not be found. Try to specify it's location with `-DLIBSND_ROOT_DIR`.")
  endif()
  message(STATUS "Found libsnd: ${libsnd_LIBS}")
  list(APPEND DALI_LIBS ${libsnd_LIBS})
endif()

##################################################################
# libtar
##################################################################
if(BUILD_LIBTAR)
  find_library(libtar_LIBS
          NAMES libtar.a tar libtar
          PATHS ${LIBTAR_ROOT_DIR} "/usr/local" ${CMAKE_SYSTEM_PREFIX_PATH}
          PATH_SUFFIXES lib lib64)
  if(${libtar_LIBS} STREQUAL libtar_LIBS-NOTFOUND)
    message(FATAL_ERROR "libtar could not be found. Try to specify it's location with `-DLIBTAR_ROOT_DIR`.")
  endif()
  message(STATUS "Found libtar: ${libtar_LIBS}")
  list(APPEND DALI_LIBS ${libtar_LIBS})
  list(APPEND DALI_EXCLUDES libtar.a)
endif()


##################################################################
# nvcomp
##################################################################
set(DALI_INSTALL_REQUIRES_NVCOMP "")
if(BUILD_NVCOMP)
  if (NOT WITH_DYNAMIC_NVCOMP)
    find_library(
      nvcomp_LIBS
      NAMES nvcomp
      PATHS ${NVCOMP_ROOT_DIR} "/usr/local/cuda" "/usr/local" ${CMAKE_SYSTEM_PREFIX_PATH}
      PATH_SUFFIXES lib lib64)
    find_path(
      nvcomp_INCLUDE_DIR
      NAMES nvcomp
      PATHS ${NVCOMP_ROOT_DIR} "/usr/local/cuda" "/usr/local" ${CMAKE_SYSTEM_PREFIX_PATH}
      PATH_SUFFIXES include)
    if(${nvcomp_LIBS} STREQUAL nvcomp_LIBS-NOTFOUND)
      message(FATAL_ERROR "nvCOMP libs could not be found. Try to specify nvcomp location with `-DNVCOMP_ROOT_DIR`.")
    endif()
    if (${nvcomp_INCLUDE_DIR} STREQUAL nvcomp_INCLUDE_DIR-NOTFOUND)
      message(FATAL_ERROR "nvCOMP headers could not be found. Try to specify nvcomp location with `-DNVCOMP_ROOT_DIR`.")
    endif()
    message(STATUS "Found nvCOMP: ${nvcomp_LIBS} ${nvcomp_INCLUDE_DIR}.")
    include_directories(SYSTEM ${nvcomp_INCLUDE_DIR})
    list(APPEND DALI_LIBS ${nvcomp_LIBS})
  else()
    set(DALI_INSTALL_REQUIRES_NVCOMP "\'nvidia-nvcomp-cu${CUDA_VERSION_MAJOR} == 5.0.0.6\',")
    message(STATUS "Adding nvComp requirement as: ${DALI_INSTALL_REQUIRES_NVCOMP}")
  endif()
endif()


##################################################################
# FFmpeg
##################################################################
if(BUILD_FFMPEG)
  include(CheckStructHasMember)
  include(CheckTypeSize)

  set(FFMPEG_ROOT_DIR "" CACHE PATH "Folder contains ffmpeg")
  set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH YES)

  find_package(PkgConfig REQUIRED)
  foreach(m avformat avcodec avfilter avutil swscale)
      # We do a find_library only if FFMPEG_ROOT_DIR is provided
      if(NOT FFMPEG_ROOT_DIR)
        string(TOUPPER ${m} M)
        pkg_check_modules(${m} REQUIRED lib${m})
        list(APPEND FFmpeg_LIBS ${m})
      else()
        # Set the name of the destination variable, it cannot be the same across
        # consecutive find_library calls to avoid caching
        set(FFmpeg_Lib "FFmpeg_Lib${m}")
        find_library(${FFmpeg_Lib} ${m}
              PATHS ${FFMPEG_ROOT_DIR}
              PATH_SUFFIXES lib lib64
              NO_DEFAULT_PATH)
        list(APPEND FFmpeg_LIBS ${${FFmpeg_Lib}})
      endif()
  endforeach(m)

  include_directories(${avformat_INCLUDE_DIRS})
  list(APPEND DALI_LIBS ${avformat_LIBRARIES})
  CHECK_STRUCT_HAS_MEMBER("struct AVStream" codecpar libavformat/avformat.h HAVE_AVSTREAM_CODECPAR LANGUAGE CXX)
  set(CMAKE_EXTRA_INCLUDE_FILES libavcodec/avcodec.h)
  CHECK_TYPE_SIZE("AVBSFContext" AVBSFCONTEXT LANGUAGE CXX)

  list(APPEND DALI_LIBS ${FFmpeg_LIBS})
endif()

##################################################################
# Boost preprocessor
##################################################################
include_directories(${PROJECT_SOURCE_DIR}/third_party/boost/preprocessor/include)

##################################################################
# RapidJSON
##################################################################
include_directories(${PROJECT_SOURCE_DIR}/third_party/rapidjson/include)

##################################################################
# FFTS
##################################################################
if (BUILD_FFTS)
  set(GENERATE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "-fPIC")
  set(ENABLE_SHARED OFF CACHE BOOL "shared library target")
  set(ENABLE_STATIC ON CACHE BOOL "static library target")
  # dynamic machine code generation works only for x86
  if(NOT (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64"))
    set(DISABLE_DYNAMIC_CODE ON CACHE BOOL "Disables the use of dynamic machine code generation")
  endif()

  # Workaround for Clang as msse3 is only enabled if GCC is detected
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse -msse2 -msse3")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -msse -Xcompiler -msse2 -Xcompiler -msse3")
  endif()

  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/third_party/ffts EXCLUDE_FROM_ALL)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/ffts/include)
  list(APPEND DALI_LIBS ffts)
  list(APPEND DALI_EXCLUDES libffts.a)
endif()

##################################################################
# CUTLASS
##################################################################
include_directories(SYSTEM third_party/cutlass/include)
include_directories(SYSTEM third_party/cutlass/tools/util/include)

##################################################################
# CocoAPI
##################################################################
set(SOURCE_FILES third_party/cocoapi/common/maskApi.c)
add_library(cocoapi STATIC ${SOURCE_FILES})
set_target_properties(cocoapi PROPERTIES POSITION_INDEPENDENT_CODE ON)
list(APPEND DALI_LIBS cocoapi)
list(APPEND DALI_EXCLUDES libcocoapi.a)

##################################################################
# cfitsio
##################################################################
if(BUILD_CFITSIO)
  find_library(cfitsio_LIBS
          NAMES libcfitsio.so libcfitsio
          PATHS ${CFITSIO_ROOT_DIR} "/usr/local" ${CMAKE_SYSTEM_PREFIX_PATH}
          PATH_SUFFIXES lib lib64)
  if(${cfitsio_LIBS} STREQUAL cfitsio_LIBS-NOTFOUND)
    message(FATAL_ERROR "cfitsio could not be found. Try to specify it's location with `-DCFITSIO_ROOT_DIR`.")
  endif()
  message(STATUS "Found cfitsio: ${cfitsio_LIBS}")
  list(APPEND DALI_LIBS ${cfitsio_LIBS})
endif()

##################################################################
# CV-CUDA
##################################################################
if (BUILD_CVCUDA)
  set(DALI_BUILD_PYTHON ${BUILD_PYTHON})
  set(BUILD_PYTHON OFF)
  # for now we use only median blur from CV-CUDA
  set(CV_CUDA_SRC_PATERN medianblur median_blur morphology warp HQResize)
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/third_party/cvcuda)
  set(BUILD_PYTHON ${DALI_BUILD_PYTHON})
endif()

##################################################################
# nvimagecodec
##################################################################
set(DALI_INSTALL_REQUIRES_NVIMGCODEC "")
if(BUILD_NVIMAGECODEC)
  set(NVIMGCODEC_MIN_VERSION "0.6.0")
  set(NVIMGCODEC_MAX_VERSION "0.7.0")
  message(STATUS "nvImageCodec - requires version >=${NVIMGCODEC_MIN_VERSION}, <${NVIMGCODEC_MAX_VERSION}")
  if (WITH_DYNAMIC_NVIMGCODEC)
    message(STATUS "nvImageCodec - dynamic load")

    # Silence DOWNLOAD_EXTRACT_TIMESTAMP warning in CMake 3.24:
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
      cmake_policy(SET CMP0135 NEW)
    endif()

    find_package(nvimgcodec ${NVIMGCODEC_MIN_VERSION}...<${NVIMGCODEC_MAX_VERSION})
    if (NOT nvimgcodec_FOUND)
      message(STATUS "nvImageCodec - not found; downloading from nvidia.com")
      # Note: We are getting the x86_64 tarball, but we are only interested in the headers.
      include(FetchContent)
      FetchContent_Declare(
        nvimgcodec_headers
        URL      https://developer.download.nvidia.com/compute/nvimgcodec/redist/nvimgcodec/linux-x86_64/nvimgcodec-linux-x86_64-0.6.0.32-archive.tar.xz
        URL_HASH SHA512=a7c894d38c78fd6fb4e460c5aebabaf90af20462faf84dcbaa310ca4842638cccd8d9628cafda1a970f865afe44815d718f65fe12f6c84160b8cd2d8485e81ca
      )
      FetchContent_Populate(nvimgcodec_headers)
      set(nvimgcodec_INCLUDE_DIR "${nvimgcodec_headers_SOURCE_DIR}/${CUDA_VERSION_MAJOR}/include")
      if (NOT EXISTS "${nvimgcodec_INCLUDE_DIR}/nvimgcodec.h")
        message(FATAL_ERROR "nvimgcodec.h not found in ${nvimgcodec_INCLUDE_DIR} - something went wrong with the download")
      endif()
    endif()
    message(STATUS "Using nvimgcodec_INCLUDE_DIR=${nvimgcodec_INCLUDE_DIR}")
    include_directories(SYSTEM ${nvimgcodec_INCLUDE_DIR})

    # Setting default installation path for dynamic loading
    message(STATUS "NVIMGCODEC_DEFAULT_INSTALL_PATH=${NVIMGCODEC_DEFAULT_INSTALL_PATH}")
    add_definitions(-DNVIMGCODEC_DEFAULT_INSTALL_PATH=\"${NVIMGCODEC_DEFAULT_INSTALL_PATH}\")

    if("$ENV{ARCH}" STREQUAL "aarch64-linux")
      message(STATUS "ARCH is set to aarch64-linux")
      set(NVIMGCODEC_PACKAGE_NAME "nvidia-nvimgcodec-tegra-cu${CUDA_VERSION_MAJOR}[all]")
      set(DALI_INSTALL_REQUIRES_NVIMGCODEC "")
    else()
      message(STATUS "ARCH is set to $ENV{ARCH}")
      set(NVIMGCODEC_PACKAGE_NAME "nvidia-nvimgcodec-cu${CUDA_VERSION_MAJOR}[all]")
      set(DALI_INSTALL_REQUIRES_NVIMGCODEC "\'${NVIMGCODEC_PACKAGE_NAME} >= ${NVIMGCODEC_MIN_VERSION}, < ${NVIMGCODEC_MAX_VERSION}',")
      message(STATUS "Adding nvimagecodec requirement as: ${DALI_INSTALL_REQUIRES_NVIMGCODEC}")
    endif()
  else()
    message(STATUS "nvImageCodec - static link")

    set(NVIMGCODEC_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/nvimgcodec")

    set(EXTRA_CMAKE_OPTIONS $ENV{EXTRA_CMAKE_OPTIONS})
    if (NOT "${EXTRA_CMAKE_OPTIONS}" STREQUAL "")
      string(REPLACE " " ";" EXTRA_CMAKE_OPTIONS_LIST ${EXTRA_CMAKE_OPTIONS})
    else()
      set(EXTRA_CMAKE_OPTIONS_LIST "")
    endif()

    include(ExternalProject)
    ExternalProject_Add(
      nvImageCodec
      GIT_REPOSITORY    https://github.com/NVIDIA/nvImageCodec.git
      GIT_TAG           v0.6.0
      GIT_SUBMODULES    "external/pybind11"
                        "external/NVTX"
                        "external/googletest"
                        "external/dlpack"
                        "external/boost/preprocessor"
      CMAKE_ARGS        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                        "-DCMAKE_INSTALL_PREFIX=${NVIMGCODEC_INSTALL_PREFIX}"
                        "-DBUILD_TEST=OFF"
                        "-DBUILD_SAMPLES=OFF"
                        "-DBUILD_PYTHON=OFF"
                        "-DBUILD_NVJPEG2K_EXT=${BUILD_NVJPEG2K}"
                        "-DWITH_DYNAMIC_NVJPEG2K=OFF"
                        "-DBUILD_NVJPEG_EXT=${BUILD_NVJPEG}"
                        "-DWITH_DYNAMIC_NVJPEG=${WITH_DYNAMIC_NVJPEG}"
                        "-DBUILD_NVTIFF_EXT=OFF"
                        "-DWITH_DYNAMIC_NVTIFF=OFF"
                        "-DBUILD_NVBMP_EXT=OFF"
                        "-DBUILD_NVPNM_EXT=OFF"
                        "-DBUILD_LIBJPEG_TURBO_EXT=${BUILD_LIBJPEG_TURBO}"
                        "-DBUILD_LIBTIFF_EXT=${BUILD_LIBTIFF}"
                        "-DBUILD_OPENCV_EXT=${BUILD_OPENCV}"
                        "-DBUILD_DOCS=OFF"
                        "${EXTRA_CMAKE_OPTIONS_LIST}"
      PREFIX            "${NVIMGCODEC_INSTALL_PREFIX}"
    )
    set(nvimgcodec_INCLUDE_DIR "${NVIMGCODEC_INSTALL_PREFIX}/include")
    set(nvimgcodec_LIBRARY_DIR "${NVIMGCODEC_INSTALL_PREFIX}/lib64")
    message(STATUS "Using nvimgcodec_INCLUDE_DIR=${nvimgcodec_INCLUDE_DIR}")
    message(STATUS "Using nvimgcodec_LIBRARY_DIR=${nvimgcodec_LIBRARY_DIR}")
    include_directories(SYSTEM ${nvimgcodec_INCLUDE_DIR})
    link_directories(${nvimgcodec_LIBRARY_DIR})

    set(NVIMGCODEC_LIBS "")
    list(APPEND NVIMGCODEC_LIBS nvimgcodec_static)
    list(APPEND DALI_EXCLUDES libnvimgcodec_static.a)

    list(APPEND NVIMGCODEC_LIBS opencv_ext_static)
    list(APPEND DALI_EXCLUDES libopencv_ext_static.a)

    if (BUILD_LIBJPEG_TURBO)
      message(STATUS "nvImageCodec - Include libjpeg-turbo extension")
      list(APPEND NVIMGCODEC_LIBS jpeg_turbo_ext_static)
      list(APPEND DALI_EXCLUDES libjpeg_turbo_ext_static.a)
      endif()
    if (BUILD_LIBTIFF)
      message(STATUS "nvImageCodec - Include libtiff extension")
      list(APPEND NVIMGCODEC_LIBS tiff_ext_static)
      list(APPEND DALI_EXCLUDES libtiff_ext_static.a)
      endif()
    if (BUILD_NVJPEG2K)
      message(STATUS "nvImageCodec - Include nvjpeg2k extension")
      list(APPEND NVIMGCODEC_LIBS nvjpeg2k_ext_static)
      list(APPEND DALI_EXCLUDES libnvjpeg2k_ext_static.a)
      endif()
    if (BUILD_NVJPEG)
      message(STATUS "nvImageCodec - Include nvjpeg extension")
      list(APPEND NVIMGCODEC_LIBS nvjpeg_ext_static)
      list(APPEND DALI_EXCLUDES libnvjpeg_ext_static.a)
      endif()
  endif()
endif()


##################################################################
# AWS SDK
##################################################################
if(BUILD_AWSSDK)
  find_path(AWSSDK_INCLUDE_DIR aws/core/Aws.h)
  find_library(AWS_CPP_SDK_CORE_LIB NAMES aws-cpp-sdk-core)
  find_library(AWS_CPP_SDK_S3_LIB NAMES aws-cpp-sdk-s3)
  if ("${AWSSDK_INCLUDE_DIR}" STREQUAL "AWSSDK_INCLUDE_DIR-NOTFOUND" OR
      "${AWS_CPP_SDK_CORE_LIB}" STREQUAL "AWS_CPP_SDK_CORE_LIB-NOTFOUND" OR
      "${AWS_CPP_SDK_S3_LIB}" STREQUAL "AWS_CPP_SDK_S3_LIB-NOTFOUND")
      message(WARNING "AWS SDK not found. Disabling AWS SDK support.")
      set(BUILD_AWSSDK OFF)
  else()
    set(AWSSDK_LIBRARIES "")
    list(APPEND AWSSDK_LIBRARIES ${AWS_CPP_SDK_S3_LIB})
    list(APPEND AWSSDK_LIBRARIES ${AWS_CPP_SDK_CORE_LIB})
    message(STATUS "AWSSDK_INCLUDE_DIR=${AWSSDK_INCLUDE_DIR}")
    message(STATUS "AWSSDK_LIBRARIES=${AWSSDK_LIBRARIES}")
  endif()
endif()
