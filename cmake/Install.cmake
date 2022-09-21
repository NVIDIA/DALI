# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if (BUILD_DALI_IMGCODEC)
  list(APPEND DALI_LIBS dali_imgcodec)
endif()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DALI/DALIConfigVersion.cmake"
  VERSION ${DALI_VERSION}
  COMPATIBILITY SameMajorVersion
)

set(DALI_CONFIG_PACKAGE_LOCATION ${CMAKE_INSTALL_LIBDIR}/cmake/DALI CACHE STRING
    "Installation directory for cmake files, a relative path that will be joined with ${CMAKE_INSTALL_PREFIX} or an absolute path.")

# Note the '/' at the end of first path and not present at the end of the other
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        COMPONENT Devel
        DESTINATION include
        FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.hpp")

install(DIRECTORY ${CMAKE_SOURCE_DIR}/dali
        COMPONENT Devel-Private
        DESTINATION include
        FILES_MATCHING
          PATTERN "*_test.h" EXCLUDE
          PATTERN "*.h")

install(TARGETS ${DALI_LIBS}
        EXPORT DALITargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION ${DALI_INC_DIR}
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)

export(EXPORT DALITargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/DALI/DALITargets.cmake"
  NAMESPACE DALI::
)

configure_file(cmake/DALIConfig.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/DALI/DALIConfig.cmake"
  COPYONLY
)

install(EXPORT DALITargets
  FILE DALITargets.cmake
  NAMESPACE DALI::
  DESTINATION ${DALI_CONFIG_PACKAGE_LOCATION}
  COMPONENT Devel
)

install(
  FILES
    cmake/DALIConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/DALI/DALIConfigVersion.cmake"
  DESTINATION "${DALI_CONFIG_PACKAGE_LOCATION}"
  COMPONENT Devel
)
