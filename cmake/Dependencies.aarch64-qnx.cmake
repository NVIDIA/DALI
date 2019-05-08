# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

# Configurations for aarch64-qnx
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

find_package(CUDA 10.0 REQUIRED)

set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_HOST})
set(CUDA_TOOLKIT_TARGET_DIR ${CUDA_TARGET})

find_library(CUDA_CUDART_LIBRARY cudart
  PATHS ${CUDA_TOOLKIT_TARGET_DIR}
  PATH_SUFFIXES lib64 lib)

message(STATUS "Found cudart at ${CUDA_CUDART_LIBRARY}")

set(CUDA_TOOLKIT_ROOT_DIR_INTERNAL ${CUDA_TOOLKIT_ROOT_DIR} CACHE PATH "Cuda toolkit internal root location")
set(CUDA_TOOLKIT_TARGET_DIR_INTERNAL ${CUDA_TOOLKIT_TARGET_DIR} CACHE PATH "Cuda toolkit target location")

set(CUDA_LIBRARIES ${CUDA_TOOLKIT_TARGET_DIR}/lib)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "QNX does not have librt.so")

# Check QNX
if(NOT DEFINED ENV{QNX_TARGET} OR NOT DEFINED ENV{QNX_HOST})
  message("QNX_TARGET and QNX_HOST not exported")
endif()

set(CMAKE_SYSROOT $ENV{QNX_TARGET})

if(NOT IS_DIRECTORY ${CMAKE_SYSROOT}/aarch64le)
  message(FATAL_ERROR"[ERROR] Please set $QNX_TARGET like below.\n $ export QNX_TARGET=/PATH/TO/qnx700/target/qnx7\n")
endif()

if(NOT EXISTS ${CMAKE_C_COMPILER})
  message(FATAL_ERROR "[ERROR] Please set $QNX_HOST like below.\n $ export QNX_HOST=/ PATH/TO/qnx700/host/linux/x86_64\n")
endif()

list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libcudart.so)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libnppc_static.a)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libnppicom_static.a)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libnppicc_static.a)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libnppig_static.a)

include_directories(${CUDA_TOOLKIT_TARGET_DIR}/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CUDA_LIBRARIES} -L${CUDA_LIBRARIES}/stubs -lcudart -lnppc_static -lnppicom_static -lnppicc_static -lnppig_static -lnpps -lnppc -lculibos")

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
  find_package(JPEG 62 REQUIRED) # 1.5.3 version
  include_directories(SYSTEM ${JPEG_INCLUDE_DIR})
  message("Using libjpeg-turbo at ${JPEG_LIBRARY}")
  list(APPEND DALI_LIBS ${JPEG_LIBRARY})
  add_definitions(-DDALI_USE_JPEG_TURBO)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ljpeg -lturbojpeg")
else()
  # Note: Support for disabling libjpeg-turbo is unofficial
  message(STATUS "Building WITHOUT JpegTurbo")
endif()

##################################################################
# OpenCV
##################################################################

# Path to architecture specific opencv
if(NOT DEFINED OPENCV_PATH)
  message("OpenCV path not exported for architecture configured")
endif()

set(OpenCV_INCLUDE_DIRS ${OPENCV_PATH}/include)
set(OpenCV_LIBRARIES ${OPENCV_PATH}/lib)

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${OpenCV_LIBRARIES}/libopencv_core.so)
list(APPEND DALI_LIBS ${OpenCV_LIBRARIES}/libopencv_imgproc.so)
list(APPEND DALI_LIBS ${OpenCV_LIBRARIES}/libopencv_imgcodecs.so)


##################################################################
# Protobuf
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

add_definitions(-DGOOGLE_PROTOBUF_ARCH_64_BIT)
add_definitions(-DGOOGLE_PROTOBUF_ARCH_X64)

set(PROTO_LIB_PATH ${PROTOBUF_TARGET}/lib)

list(APPEND DALI_LIBS ${PROTO_LIB_PATH}/libprotobuf.so)
list(APPEND DALI_LIBS ${PROTO_LIB_PATH}/libprotobuf-lite.so)
list(APPEND DALI_LIBS ${PROTO_LIB_PATH}/libprotoc.so)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${PROTO_LIB_PATH} -lprotobuf -lprotobuf-lite -lprotoc")

###################################################################
# ffmpeg
###################################################################
include(CheckStructHasMember)
include(CheckTypeSize)

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
CHECK_STRUCT_HAS_MEMBER("struct AVStream" codecpar libavformat/avformati.h  HAVE_AVSTREAM_CODECPAR LANGUAGE C)
set(CMAKE_EXTRA_INCLUDE_FILES libavcodec/avcodec.h)
CHECK_TYPE_SIZE("AVBSFContext" AVBSFCONTEXT LANGUAGE CXX)

list(APPEND DALI_LIBS ${FFmpeg_LIBS})
