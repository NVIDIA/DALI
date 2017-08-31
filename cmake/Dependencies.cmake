# For CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND NDLL_LIBS ${CUDA_LIBRARIES})

# Google C++ testing framework
if (BUILD_TEST)
  set(BUILD_GTEST ON CACHE INTERNAL "Builds gtest submodule")
  set(BUILD_GMOCK OFF CACHE INTERNAL "Builds gmock submodule")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
endif()

# LibJpegTurbo
find_package(JpegTurbo)
include_directories(${JPEG_TURBO_INCLUDE_DIR})
list(APPEND NDLL_LIBS ${JPEG_TURBO_LIBRARY})
