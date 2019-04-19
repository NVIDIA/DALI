# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
# libjpeg-turbo
##################################################################
if (BUILD_JPEG_TURBO)
  find_package(JPEG 62 REQUIRED) # 1.5.3 version
  include_directories(SYSTEM ${JPEG_INCLUDE_DIR})
  message("Using libjpeg-turbo at ${JPEG_LIBRARY}")
  list(APPEND DALI_LIBS ${JPEG_LIBRARY})
  add_definitions(-DDALI_USE_JPEG_TURBO)
else()
  # Note: Support for disabling libjpeg-turbo is unofficial
  message(STATUS "Building WITHOUT JpegTurbo")
endif()

##################################################################
# PyBind
##################################################################
if (BUILD_PYTHON)
  set(PYBIND11_CPP_STANDARD -std=c++11)
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/third_party/pybind11)
else()
  message(STATUS "Building WITHOUT Python bindings")
endif()

##################################################################
# LMDB
##################################################################
if (BUILD_LMDB)
  find_package(LMDB 0.9 REQUIRED)
  include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
  list(APPEND DALI_LIBS ${LMDB_LIBRARIES})
  list(APPEND DALI_EXCLUDES liblmdb.a)
else()
  message(STATUS "Building WITHOUT LMDB support")
endif()

##################################################################
# FFmpeg
##################################################################

if(BUILD_FFMPEG)
  include(CheckStructHasMember)
  include(CheckTypeSize)

  set(FFMPEG_ROOT_DIR "" CACHE PATH "Folder contains FFmeg")
  set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH YES)

  find_package(PkgConfig REQUIRED)
  foreach(m avformat avcodec avfilter avutil)
      # We do a find_library only if FFMPEG_ROOT_DIR is provided
      if(NOT FFMPEG_ROOT_DIR)
        string(TOUPPER ${m} M)
        pkg_check_modules(${m} REQUIRED lib${m})
        list(APPEND FFmpeg_LIBS ${m})
      else()
        find_library(FFmpeg_Lib ${m}
              PATHS ${FFMPEG_ROOT_DIR}
              PATH_SUFFIXES lib lib64
              NO_DEFAULT_PATH)
        list(APPEND FFmpeg_LIBS ${FFmpeg_Lib})
        message(STATUS ${m})
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
# Boost prerocessor
##################################################################
include_directories(${PROJECT_SOURCE_DIR}/third_party/boost/preprocessor/include)

##################################################################
# RapidJSON
##################################################################
include_directories(${PROJECT_SOURCE_DIR}/third_party/rapidjson/include)