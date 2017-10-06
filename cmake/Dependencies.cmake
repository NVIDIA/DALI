# For CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND NDLL_LIBS ${CUDA_LIBRARIES})
  
# TODO(tgale): Is there a way to automate this and not hack
# in the path off the base CUDA install?
list(APPEND NDLL_LIBS ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/stubs/libnvidia-ml.so)

# For NPP
find_cuda_helper_libs(nppig)
list(APPEND NDLL_LIBS ${CUDA_nppig_LIBRARY})

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
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
endif()

# Google Benchmark
if (BUILD_BENCHMARK)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "Build benchmark testsuite")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/benchmark)
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
