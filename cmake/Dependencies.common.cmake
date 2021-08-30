# Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  set(PYBIND11_CPP_STANDARD -std=c++14)
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
          PATHS ${CMAKE_SYSTEM_PREFIX_PATH} ${LIBSND_ROOT_DIR} "/usr/local"
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
          PATHS ${CMAKE_SYSTEM_PREFIX_PATH} ${LIBTAR_ROOT_DIR} "/usr/local"
          PATH_SUFFIXES lib lib64)
  if(${libtar_LIBS} STREQUAL libtar_LIBS-NOTFOUND)
    message(FATAL_ERROR "libtar could not be found. Try to specify it's location with `-DLIBTAR_ROOT_DIR`.")
  endif()
  message(STATUS "Found libtar: ${libtar_LIBS}")
  list(APPEND DALI_LIBS ${libtar_LIBS})
  list(APPEND DALI_EXCLUDES libtar.a)
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
  foreach(m avformat avcodec avfilter avutil)
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

##################################################################
# CocoAPI
##################################################################
set(SOURCE_FILES third_party/cocoapi/common/maskApi.c)
add_library(cocoapi STATIC ${SOURCE_FILES})
set_target_properties(cocoapi PROPERTIES POSITION_INDEPENDENT_CODE ON)
list(APPEND DALI_LIBS cocoapi)
list(APPEND DALI_EXCLUDES libcocoapi.a)

##################################################################
# libcu++
##################################################################
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/libcudacxx/include)
