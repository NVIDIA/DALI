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

##################################################################
# Utility function, useful when building custom C++ library where
# DALI is used as a shared library. Assumes that DALI wheel is
# installed within the system. Acquires paths to DALI libraries
# and dependencies introduced by the wheel. On the event that
# multiple DALI wheels are installed, picks the most recent one.
#
# Usage:
#
# find_dali(DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)
#
# target_include_directories(my_target ${DALI_INCLUDE_DIR})
# target_link_directories(my_target ${DALI_LIB_DIR})
# target_link_libraries(my_target ${DALI_LIBRARIES})
##################################################################
function(find_dali DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)
    execute_process(
            COMMAND python3 -c "import nvidia.dali.sysconfig as dali_sc; print(dali_sc.get_include_dir(), end='')"
            OUTPUT_VARIABLE DALI_INCLUDE_DIR_LOC
            RESULT_VARIABLE DALI_INCLUDE_DIR_RESULT
    )

    execute_process(
            COMMAND python3 -c "import nvidia.dali.sysconfig as dali_sc; print(dali_sc.get_lib_dir(), end='')"
            OUTPUT_VARIABLE DALI_LIB_DIR_LOC
            RESULT_VARIABLE DALI_LIB_DIR_RESULT
    )

    if (${DALI_INCLUDE_DIR_RESULT} EQUAL "1" OR ${DALI_LIB_DIR_RESULT} EQUAL "1")
        message(FATAL_ERROR "Error acquiring DALI. Please verify, that the DALI wheel is installed")
    endif ()

    set(${DALI_INCLUDE_DIR} ${DALI_INCLUDE_DIR_LOC} PARENT_SCOPE)
    set(${DALI_LIB_DIR} ${DALI_LIB_DIR_LOC} PARENT_SCOPE)
    set(${DALI_LIBRARIES} dali dali_operators PARENT_SCOPE)
    message(STATUS "DALI libraries DIR: " ${DALI_LIB_DIR})
    message(STATUS "DALI include DIR: " ${DALI_INCLUDE_DIR})
    message(STATUS "DALI linked libraries: " ${DALI_LIBRARIES})
endfunction()
