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

  if (${NVJPEG_LIBRARY_0_2_0})
    add_definitions(-DNVJPEG_LIBRARY_0_2_0)
  endif()

  if (${NVJPEG_DECOUPLED_API})
    add_definitions(-DNVJPEG_DECOUPLED_API)
  endif()
else()
  # Note: Support for disabling nvJPEG is unofficial
  message(STATUS "Building WITHOUT nvJPEG")
endif()

# NVIDIA NPPC library
find_cuda_helper_libs(nppc_static)

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
list(APPEND DALI_LIBS ${CUDA_nppc_static_LIBRARY})
list(APPEND DALI_EXCLUDES libnppc_static.a)

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
# OpenCV
##################################################################
# For OpenCV 3 and later, 'imdecode()' is in the imgcodecs library
find_package(OpenCV 4.0 QUIET COMPONENTS core imgproc imgcodecs)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3.0 QUIET COMPONENTS core imgproc imgcodecs)
endif()
if(NOT OpenCV_FOUND)
  # Note: OpenCV 2 support is unofficial
  # For OpenCV 2.x, image encode/decode functions are in highgui
  find_package(OpenCV 2.0 REQUIRED COMPONENTS core imgproc highgui)
endif()
message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} (found suitable version \"${OpenCV_VERSION}\", minimum required is \"2.0\")")
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${OpenCV_LIBRARIES})
message("OpenCV libraries: ${OpenCV_LIBRARIES}")
list(APPEND DALI_EXCLUDES libopencv_core.a;libopencv_imgproc.a;libopencv_highgui.a;libopencv_imgcodecs.a;
                          liblibwebp.a;libittnotify.a;libpng.a;liblibtiff.a;liblibjasper.a;libIlmImf.a;
                          liblibjpeg-turbo.a)

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
# link statically
set(Protobuf_USE_STATIC_LIBS "ON")
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
# hide things from the protobuf, all we export is only is API generated from our proto files
list(APPEND DALI_EXCLUDES libprotobuf.a)


##################################################################
# FFmpeg
##################################################################

include(CheckStructHasMember)
include(CheckTypeSize)

set(FFMPEG_ROOT_DIR "" CACHE PATH "Folder contains FFmeg")

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
CHECK_STRUCT_HAS_MEMBER("struct AVStream" codecpar libavformat/avformat.h HAVE_AVSTREAM_CODECPAR LANGUAGE C)
set(CMAKE_EXTRA_INCLUDE_FILES libavcodec/avcodec.h)
CHECK_TYPE_SIZE("AVBSFContext" AVBSFCONTEXT LANGUAGE CXX)

list(APPEND DALI_LIBS ${FFmpeg_LIBS})

##################################################################
# Exclude stdlib
##################################################################
list(APPEND DALI_EXCLUDES libsupc++.a;libstdc++.a;libstdc++_nonshared.a;)


##################################################################
# Turing Optical flow API
##################################################################
include_directories(${PROJECT_SOURCE_DIR}/third_party/turing_of)