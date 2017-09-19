# For CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND HYBRID_DECODE_LIBS ${CUDA_LIBRARIES})

# For NPP: This is a hack to build against the npp static libs (until the
# shared libs get fixed and actually have the functions we need in them)
list(APPEND HYBRID_DECODE_LIBS
  "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppicom_static.a"
  "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppicc_static.a"
  "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppc_static.a"
  "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a"
  )

# Google C++ testing framework
# HACK: build against the gtest build done by ndll
# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIRS})
# list(APPEND HYBRID_DECODE_LIBS ${GTEST_LIBRARIES})
