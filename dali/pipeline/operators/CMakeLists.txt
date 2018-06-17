# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
add_subdirectory(color)
add_subdirectory(decoder)
add_subdirectory(displacement)
add_subdirectory(fused)
add_subdirectory(reader)
add_subdirectory(resize)
add_subdirectory(support)
add_subdirectory(util)

# Get all the source files and dump test files
file(GLOB tmp *.cc *.cu)
set(DALI_SRCS ${DALI_SRCS} ${tmp})
file(GLOB tmp *_test.cc)
remove(DALI_SRCS "${DALI_SRCS}" ${tmp})

set(DALI_SRCS ${DALI_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  file(GLOB tmp *_test.cc)
  set(DALI_TEST_SRCS ${DALI_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()
