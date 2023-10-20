# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
if(BUILD_NVCOMP)
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
  set(CV_CUDA_SRC_PATERN medianblur median_blur)
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/third_party/cvcuda)
  list(APPEND DALI_LIBS cvcuda nvcv_types)
  set(BUILD_PYTHON ${DALI_BUILD_PYTHON})
endif()
