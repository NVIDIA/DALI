# For CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND NDLL_LIBS ${CUDA_LIBRARIES})

# For NPP
find_cuda_helper_libs(nppig)
list(APPEND NDLL_LIBS ${CUDA_nppig_LIBRARY})

# For NVML
find_library(CUDA_NVML_LIB nvidia-ml
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/stubs targets/x86_64-linux/lib/stubs)
if (CUDA_NVML_LIB)
  message(STATUS "Found libnvidia-ml: ${CUDA_NVML_LIB}")
  list(APPEND NDLL_LIBS ${CUDA_NVML_LIB})
else()
  message(FATAL_ERROR "Cannot find libnvidia-ml.so")
endif()

# NVTX for profiling
if (USE_NVTX)
  find_cuda_helper_libs(nvToolsExt)
  list(APPEND NDLL_LIBS ${CUDA_nvToolsExt_LIBRARY})
  add_definitions(-DNDLL_USE_NVTX)
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
find_package(JpegTurbo REQUIRED)
include_directories(SYSTEM ${JPEG_TURBO_INCLUDE_DIR})
list(APPEND NDLL_LIBS ${JPEG_TURBO_LIBRARY})

# OpenCV
# Note: OpenCV 3.* 'imdeocde()' is in the imgcodecs library
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
if (OpenCV_FOUND)
  string(SUBSTRING ${OpenCV_VERSION} 0 1 OCV_VERSION)
  if ("${OCV_VERSION}" STREQUAL "3")
    # Get the imgcodecs library
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs)
  endif()

  # Check for opencv
  message(STATUS "Found OpenCV ${OpenCV_VERSION} (libs: ${OpenCV_LIBRARIES})")
endif()
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND NDLL_LIBS ${OpenCV_LIBRARIES})

# Hybrid Decode
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/hybrid_decode)
include_directories(${PROJECT_SOURCE_DIR}/third_party/hybrid_decode/include)

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
    list(APPEND NDLL_LIBS ${LMDB_LIBRARIES})
  else()
    message(STATUS "LMDB not found")
  endif()
endif()

# protobuf
find_package(Protobuf REQUIRED)

if (PROTOBUF_FOUND)
  message(STATUS "Found Protobuf ${PROTOBUF_INCLUDE_DIRS} : ${PROTOBUF_LIBRARY}")
  include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIRS})
  list(APPEND NDLL_LIBS ${PROTOBUF_LIBRARY})
else()
  message(STATUS "Protobuf not found")
endif()
