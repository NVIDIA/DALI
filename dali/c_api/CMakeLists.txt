# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
# Get all the source files
file(GLOB tmp *.cc)
set(DALI_SRCS ${DALI_SRCS} ${tmp} PARENT_SCOPE)
