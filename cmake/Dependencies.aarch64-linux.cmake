#############################
#        CUDA TOOLKIT
#############################

find_package(CUDA 10.0 REQUIRED)

set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_HOST})
set(CUDA_TOOLKIT_TARGET_DIR ${CUDA_TARGET})
set(CUDA_TOOLKIT_ROOT_DIR_INTERNAL ${CUDA_TOOLKIT_ROOT_DIR})
set(CUDA_TOOLKIT_TARGET_DIR_INTERNAL ${CUDA_TOOLKIT_TARGET_DIR})
set(CUDA_LIBRARIES ${CUDA_TOOLKIT_TARGET_DIR}/lib)

list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libcudart.so)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libnppc_static.a)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libnppicom_static.a)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libnppicc_static.a)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libnppig_static.a)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libculibos.a)
list(APPEND DALI_LIBS ${CMAKE_DL_LIBS})

list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/stubs/libnppc.so)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/stubs/libnppicom.so)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/stubs/libnppicc.so)
list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/stubs/libnppig.so)

include_directories(${CUDA_TOOLKIT_TARGET_DIR}/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CUDA_LIBRARIES} -L${CUDA_LIBRARIES}/stubs -lcudart -lnppc_static -lnppicom_static -lnppicc_static -lnppig_static -lnpps -lnppc -lculibos")

# NVTX for profiling
if (BUILD_NVTX)
  list(APPEND DALI_LIBS ${CUDA_LIBRARIES}/libnvToolsExt.so)
  add_definitions(-DDALI_USE_NVTX)
endif()


##############################
#         OPENCV
##############################

# Path to architecture specific opencv
if(NOT DEFINED OPENCV_PATH)
  message("OpenCV path not exported for architecture configured")
endif()

message(STATUS "Found OpenCV at ${OPENCV_PATH}")

set(OpenCV_INCLUDE_DIRS ${OPENCV_PATH}/include)
set(OpenCV_LIBRARIES ${OPENCV_PATH}/lib)

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
message("OPENCV check for ${OpenCV_LIBRARIES} and include dir ${OpenCV_INCLUDE_DIRS}")
list(APPEND DALI_LIBS ${OpenCV_LIBRARIES}/libopencv_core.so)
list(APPEND DALI_LIBS ${OpenCV_LIBRARIES}/libopencv_imgproc.so)
list(APPEND DALI_LIBS ${OpenCV_LIBRARIES}/libopencv_imgcodecs.so)

##################################################################
# PyBind
##################################################################
if (BUILD_PYTHON)
  set(PYBIND11_CPP_STANDARD -std=c++11)
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/third_party/pybind11)
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

add_definitions(-DGOOGLE_PROTOBUF_ARCH_64_BIT)
add_definitions(-D__aarch64__)

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
CHECK_STRUCT_HAS_MEMBER("struct AVStream" codecpar libavformat/avformat.h HAVE_AVSTREAM_CODECPAR LANGUAGE C)
set(CMAKE_EXTRA_INCLUDE_FILES libavcodec/avcodec.h)
CHECK_TYPE_SIZE("AVBSFContext" AVBSFCONTEXT LANGUAGE CXX)

list(APPEND DALI_LIBS ${FFmpeg_LIBS})

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
