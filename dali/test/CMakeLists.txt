# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
if (BUILD_TEST)
  # get all the test srcs
  file(GLOB tmp *_test.cc *.cu)
  set(DALI_TEST_SRCS ${DALI_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()
