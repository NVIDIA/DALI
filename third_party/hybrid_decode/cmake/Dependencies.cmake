# For CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND HYBRID_DECODE_LIBS ${CUDA_LIBRARIES})

# For NPP
find_cuda_helper_libs(nppicom)
find_cuda_helper_libs(nppicc)
find_cuda_helper_libs(nppc)
list(APPEND HYBRID_DECODE_LIBS
  ${CUDA_nppicom_LIBRARY}
  ${CUDA_nppicc_LIBRARY}
  ${CUDA_nppc_LIBRARY}
  )

# Compile w/ NVTX
if (USE_NVTX)
  # message(STATUS "COMPILING HYBDEC W/ TIMERANGES")
  # find_cuda_helper_libs(nvToolsExt)
  # list(APPEND HYBRID_DECODE_LIBS ${CUDA_nvToolsExt_LIBRARY})
  # add_definitions(-DENABLE_TIMERANGES)
endif()

# Google C++ testing framework
# HACK: build against the gtest build done by ndll
# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIRS})
# list(APPEND HYBRID_DECODE_LIBS ${GTEST_LIBRARIES})
