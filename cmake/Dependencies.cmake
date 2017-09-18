# For CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND NDLL_LIBS ${CUDA_LIBRARIES})
message(STATUS "${CUDA_LIBRARIES}")
message(STATUS ${CUDA_TOOLKIT_ROOT_DIR})
  
# TODO(tgale): Is there a way to automate this and not hack
# in the path off the base CUDA install?
list(APPEND NDLL_LIBS ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/stubs/libnvidia-ml.so)

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
# Note: OpenCV 3.* 'imdeocde()' is in the imgcodecs library. In
# earlier versions (like the one that can be installed w/ apt-get)
# it is in the highgui library. Building OCV from source has
# failed over and over, so we're now working with the apt-get version
# find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
if (OpenCV_FOUND)
  message(STATUS "Found OpenCV (libs: ${OpenCV_LIBRARIES})")
endif()
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND NDLL_LIBS ${OpenCV_LIBRARIES})
