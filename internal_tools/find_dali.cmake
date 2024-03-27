# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

####################################################################
# Utility function, useful when building custom C++ library where
# DALI is used as a shared library. Assumes that DALI wheel is
# installed within the system. Acquires paths to DALI libraries and
# dependencies introduced by the wheel. On the event that multiple
# DALI wheels exist, picks the one that's most recently installed.
#
# Usage:
#
# find_dali(DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)
#
# target_include_directories(my_target ${DALI_INCLUDE_DIR})
# target_link_directories(my_target ${DALI_LIB_DIR})
# target_link_libraries(my_target ${DALI_LIBRARIES})
###################################################################
function(find_dali DALI_INCLUDE_DIR_VAR DALI_LIB_DIR_VAR DALI_LIBRARIES_VAR)
    execute_process(
            COMMAND python -c "import nvidia.dali as dali; print(dali.sysconfig.get_include_dir(), end='')"
            OUTPUT_VARIABLE DALI_INCLUDE_DIR
            RESULT_VARIABLE INCLUDE_DIR_RESULT)

    if (${INCLUDE_DIR_RESULT} EQUAL "1")
        message(FATAL_ERROR "Failed to get include paths for DALI. Make sure that DALI is installed.")
    endif ()

    execute_process(
            COMMAND python -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir(), end='')"
            OUTPUT_VARIABLE DALI_LIB_DIR
            RESULT_VARIABLE LIB_DIR_RESULT)

    if (${LIB_DIR_RESULT} EQUAL "1")
        message(FATAL_ERROR "Failed to get library paths for DALI. Make sure that DALI is installed.")
    endif ()

    set(${DALI_INCLUDE_DIR_VAR} ${DALI_INCLUDE_DIR} PARENT_SCOPE)
    set(${DALI_LIB_DIR_VAR} ${DALI_LIB_DIR} PARENT_SCOPE)
    set(${DALI_LIBRARIES_VAR} dali dali_kernels dali_operators PARENT_SCOPE)
    message(STATUS "DALI include DIR: " ${DALI_INCLUDE_DIR})
    message(STATUS "DALI libraries DIR: " ${DALI_LIB_DIR})
endfunction()
