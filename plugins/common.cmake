# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(PYTHON_EXECUTABLE "Python executable" OFF)
if (NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE $ENV{PYTHON_EXECUTABLE})
    if (NOT PYTHON_EXECUTABLE)
        set(PYTHON_EXECUTABLE "python")
    endif()
endif()
message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")

option(DALI_COMPILE_FLAGS "DALI compile flags" OFF)
if (NOT DALI_COMPILE_FLAGS)
    set(DALI_COMPILE_FLAGS $ENV{DALI_COMPILE_FLAGS})
    if (NOT DALI_COMPILE_FLAGS)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import nvidia.dali as dali; compile_flags=' '.join(dali.sysconfig.get_compile_flags()); print(compile_flags, end='')"
            OUTPUT_VARIABLE DALI_COMPILE_FLAGS
            RESULT_VARIABLE COMPILE_FLAGS_RESULT)

        if (${COMPILE_FLAGS_RESULT} EQUAL "1")
            message(FATAL_ERROR "Failed to get compile flags for DALI. Make sure that DALI is installed.")
        endif()
    endif()
endif()
message(STATUS "DALI_COMPILE_FLAGS=${DALI_COMPILE_FLAGS}")
set(EXTRA_COMPILE_FLAGS "${DALI_COMPILE_FLAGS} -Wall -fPIC -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_COMPILE_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_COMPILE_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${EXTRA_COMPILE_FLAGS}")

option(DALI_LIB_DIR "DALI library dir" OFF)
if (NOT DALI_LIB_DIR)
    set(DALI_LIB_DIR $ENV{DALI_LIB_DIR})
    if (NOT DALI_LIB_DIR)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir(), end='')"
            OUTPUT_VARIABLE DALI_LIB_DIR
            RESULT_VARIABLE LIB_DIR_RESULT)

        if (${LIB_DIR_RESULT} EQUAL "1")
            message(FATAL_ERROR "Failed to get library paths for DALI. Make sure that DALI is installed.")
        endif()
    endif()
endif()
message(STATUS "DALI_LIB_DIR=${DALI_LIB_DIR}")
