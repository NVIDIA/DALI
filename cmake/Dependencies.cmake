# For CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${CUDA_LIBRARIES})

# For NPP
find_cuda_helper_libs(nppicom)
find_cuda_helper_libs(nppicc)
find_cuda_helper_libs(nppc)
find_cuda_helper_libs(nppig)
list(APPEND DALI_LIBS ${CUDA_nppicom_LIBRARY}
                      ${CUDA_nppicc_LIBRARY}
                      ${CUDA_nppc_LIBRARY}
                      ${CUDA_nppig_LIBRARY})

# NVTX for profiling
if (BUILD_NVTX)
  find_cuda_helper_libs(nvToolsExt)
  list(APPEND DALI_LIBS ${CUDA_nvToolsExt_LIBRARY})
  add_definitions(-DDALI_USE_NVTX)
endif()

find_package(NVJPEG REQUIRED)
include_directories(SYSTEM ${NVJPEG_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${NVJPEG_LIBRARY})

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
find_package(JpegTurbo REQUIRED)
include_directories(SYSTEM ${JPEG_TURBO_INCLUDE_DIR})
list(APPEND DALI_LIBS ${JPEG_TURBO_LIBRARY})

# OpenCV
# Note: OpenCV 3.* 'imdecode()' is in the imgcodecs library
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
if (OpenCV_FOUND)
  string(SUBSTRING ${OpenCV_VERSION} 0 1 OCV_VERSION)
  if ("${OCV_VERSION}" STREQUAL "3")
    # Get the imgcodecs library
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  endif()

  # Check for opencv
  message(STATUS "Found OpenCV ${OpenCV_VERSION} (libs: ${OpenCV_LIBRARIES})")
endif()
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${OpenCV_LIBRARIES})

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
  else()
    message(STATUS "LMDB not found")
  endif()
endif()

# protobuf
find_package(Protobuf REQUIRED)

if (PROTOBUF_FOUND)
  # Determine if we have proto v2.x or 3.x
  execute_process(COMMAND protoc --version COMMAND cut -d " " -f 2 COMMAND cut -d . -f 1
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PROTOBUF_VERSION RESULT_VARIABLE __res)
  message(STATUS "Found Protobuf version ${PROTOBUF_VERSION} : ${PROTOBUF_INCLUDE_DIRS} : ${PROTOBUF_LIBRARY}")

  if (PROTOBUF_VERSION MATCHES 3)
    add_definitions(-DDALI_BUILD_PROTO3=1)
    set(BUILD_PROTO3 ON CACHE STRING "Build proto3")
  endif()
  include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIRS})
  list(APPEND DALI_LIBS ${PROTOBUF_LIBRARY})
else()
  message(STATUS "Protobuf not found")
endif()
