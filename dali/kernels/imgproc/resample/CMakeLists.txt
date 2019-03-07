# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project(kernels)

# Get all the source files and dump test files
file(GLOB tmp *.cc *.cu)
set(DALI_KERNEL_SRCS ${DALI_KERNEL_SRCS} ${tmp})
file(GLOB tmp *_test.cc *_test.cu)
remove(DALI_KERNEL_SRCS "${DALI_KERNEL_SRCS}" ${tmp})
set(DALI_KERNEL_SRCS ${DALI_KERNEL_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  file(GLOB tmp *_test.cc *_test.cu)
  set(DALI_KERNEL_TEST_SRCS ${DALI_KERNEL_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()
