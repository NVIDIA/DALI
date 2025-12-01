# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


########################################
#  Installing targets                  #
########################################

set (DALI_LIBS dali_core)

if (BUILD_DALI_KERNELS)
  list(APPEND DALI_LIBS dali_kernels)
endif()

if (BUILD_DALI_PIPELINE)
  list(APPEND DALI_LIBS dali)
endif()

if (BUILD_DALI_OPERATORS)
  list(APPEND DALI_LIBS dali_operators)
endif()

install(TARGETS ${DALI_LIBS}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)

# Note the '/' at the end of first path and not present at the end of the other
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/dali
        DESTINATION include
        FILES_MATCHING
          PATTERN "*_test.h" EXCLUDE
          PATTERN "*.h")
