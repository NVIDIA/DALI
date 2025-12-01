# Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES.. All rights reserved.
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


# Use CMAKE_CUDA_COMPILER to obtain the path to CUDA toolkint.
# Needed when compiling with Clang only
function(CUDA_get_toolkit_from_compiler TOOLKIT_PATH)
  get_filename_component(TOOLKIT_PATH_TMP_VAR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
  set(${TOOLKIT_PATH} ${TOOLKIT_PATH_TMP_VAR} PARENT_SCOPE)
endfunction()

# When compiling CUDA with Clang only (DALI_CLANG_ONLY=ON), we need to change the
# language properties of .cu files to allow them to use the CXX compiler (which will be Clang).
# Setting that property has narrow scope of current CMakeLists.txt, so we do this at the point
# just before creating a target.
# Clang will compile files as CUDA based on their extension.
function(adjust_source_file_language_property SOURCES)
  if (DALI_CLANG_ONLY)
    foreach(File IN LISTS SOURCES)
      if(File MATCHES ".*\.cu$")
        set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
        # CMake now forces C++ language and we need to override that to get a CUDA compilation
        set_source_files_properties(${File} PROPERTIES COMPILE_FLAGS "-x cuda")
      endif()
    endforeach()
  endif()
endfunction()



# List of currently used arch values
if (${ARCH} MATCHES "aarch64-")
  # aarch64-linux
  set(CUDA_known_archs "53" "62" "72" "75" "87" "90a")
elseif (${ARCH} MATCHES "aarch64")
  # aarch64 SBSA, only >=Volta
  # from the whole list/; "70" "75" "80" "86"
  # we pick only major arch as minor should be compatible without JITing, it should
  # shrink  the output binary
  set(CUDA_known_archs "70" "80" "90" "100" "110" "120")
else()
  # from the whole list: "35" "50" "52" "60" "61" "70" "75" "80" "86"
  # we pick only major arch as minor should be compatible without JITing, it should
  # shrink  the output binary
  set(CUDA_known_archs "35" "50" "60" "70" "80" "90" "100" "110" "120")
endif()

if ("${CUDA_VERSION_MAJOR}" EQUAL "13")
  # We need sm75 to Turing, which is still supported with cuda 13
  # but major version (sm70) is Volta, which is not supported with cuda 13."
  list(PREPEND CUDA_known_archs "75")
  list(SORT CUDA_known_archs COMPARE NATURAL)
endif()

set(CUDA_TARGET_ARCHS ${CUDA_known_archs} CACHE STRING "List of target CUDA architectures")
if ("${CUDA_TARGET_ARCHS}" STREQUAL "")
  message("CUDA_TARGET_ARCHS cannot be empty, setting to the default")
  set(CUDA_TARGET_ARCHS ${CUDA_known_archs} CACHE STRING "List of target CUDA architectures" FORCE)
endif()

# Find if passing `flags` to CUDA compiler producess success or failure
# Unix only
#
# Equivalent to dry-running preprocessing on /dev/null as .cu file
# and checking the exit code
# $ nvcc ${flags} --dryrun -E -x cu /dev/null
# or
# $ clang++ ${flags} -E -x cuda /dev/null
#
# @param out_status   TRUE iff exit code is 0, FALSE otherwise
# @param flags        flags to check
# @return out_status
function(CUDA_check_cudacc_flag out_status compiler flags)
  if (${compiler} MATCHES "clang")
    set(preprocess_empty_cu_file "-E" "-x" "cuda" "/dev/null")
  else()
    set(preprocess_empty_cu_file "--dryrun" "-E" "-x" "cu" "/dev/null")
  endif()
  set(cudacc_command ${flags} ${preprocess_empty_cu_file})
  # Run the compiler and check the exit status
  execute_process(COMMAND ${compiler} ${cudacc_command}
                  RESULT_VARIABLE tmp_out_status
                  OUTPUT_QUIET
                  ERROR_QUIET
                  )
  if (${tmp_out_status} EQUAL 0)
    set(${out_status} TRUE PARENT_SCOPE)
  else()
    set(${out_status} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Given the list of arch values, check which are supported by
#
# @param out_arch_values_allowed  List of arch values supported by the specified compiler
# @param compiler                 What compiler to use for this check
# @param arch_values_to_check     List of values to be checked against the specified compiler
#                                 for example: 60;61;70;75
# @return out_arch_values_allowed
function(CUDA_find_supported_arch_values out_arch_values_allowed compiler arch_values_to_check)
  # allow the user to pass the list like a normal variable
  set(arch_list ${arch_values_to_check} ${ARGN})
  foreach(arch IN LISTS arch_list ITEMS)
    if (${compiler} MATCHES "clang")
      CUDA_check_cudacc_flag(supported ${compiler} "--cuda-gpu-arch=sm_${arch}")
    else()
      CUDA_check_cudacc_flag(supported ${compiler} "-arch=sm_${arch}")
    endif()
    if (supported)
      set(out_list ${out_list} ${arch})
    endif()
  endforeach(arch)
  set(${out_arch_values_allowed} ${out_list} PARENT_SCOPE)
endfunction()

# Generate -gencode arch=compute_XX,code=sm_XX or --cuda-gpu-arch=sm_XX for list of supported
# arch values based on the specified compiler.
# List should be sorted in increasing order.
#
# If nvcc is used, the last arch value will be repeated as -gencode arch=compute_XX,code=compute_XX
# to ensure the generation of PTX for most recent virtual architecture
# and maintain forward compatibility.
#
# @param out_args_string  output string containing appropriate CMAKE_CUDA_FLAGS/CMAKE_CXX_FLAGS
# @param compiler         What compiler to generate flags for
# @param arch_values      list of arch values to use
# @return out_args_string
function(CUDA_get_gencode_args out_args_string compiler arch_values)
  # allow the user to pass the list like a normal variable
  set(arch_list ${arch_values} ${ARGN})
  set(out "")
  foreach(arch IN LISTS arch_list)
  if (${compiler} MATCHES "clang")
    set(out "${out} --cuda-gpu-arch=sm_${arch}")
  else()
    set(out "${out} -gencode arch=compute_${arch},code=sm_${arch}")
  endif()
  endforeach(arch)

  if (NOT ${compiler} MATCHES "clang")
    # Repeat the last one as to ensure the generation of PTX for most
    # recent virtual architecture for forward compatibility
    list(GET arch_list -1 last_arch)
    set(out "${out} -gencode arch=compute_${last_arch},code=compute_${last_arch}")
  endif()

  set(${out_args_string} ${out} PARENT_SCOPE)
endfunction()

# Generate list of xx-real for every specified supported architecture.
# List should be sorted in increasing order.
#
# The last one will also be repeated as xx-virtual to ensure the generation of PTX for most recent
# virtual architecture and maintain forward compatibility.
function(CUDA_get_cmake_cuda_archs out_args_list arch_values)
  # allow the user to pass the list like a normal variable
  set(arch_list ${arch_values} ${ARGN})
  set(out "")
  foreach(arch IN LISTS arch_list)
    set(out "${out};${arch}-real")
  endforeach(arch)

  # Repeat the last one as to ensure the generation of PTX for most
  # recent virtual architecture for forward compatibility
  list(GET arch_list -1 last_arch)
  set(out "${out};${last_arch}-virtual")

  set(${out_args_list} ${out} PARENT_SCOPE)
endfunction()


function(CUDA_find_library out_path lib_name)
    find_library(${out_path} ${lib_name} PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
                 PATH_SUFFIXES lib lib64)
endfunction()

function(CUDA_find_library_stub out_path lib_name)
    find_library(${out_path} ${lib_name} PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
                 PATH_SUFFIXES lib/stubs lib64/stubs)
endfunction()

function(CUDA_remove_toolkit_include_dirs include_dirs)
  if (NOT CMAKE_CUDA_TOOLKIT_ROOT)
    CUDA_get_toolkit_from_compiler(CUDA_TOOLKIT_PATH_VAR)
  else()
    set(CUDA_TOOLKIT_PATH_VAR ${CMAKE_CUDA_TOOLKIT_ROOT})
  endif()
  list(REMOVE_ITEM ${include_dirs} "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" "${CUDA_TOOLKIT_PATH_VAR}/include")
  set(${include_dirs} ${${include_dirs}} PARENT_SCOPE)
endfunction()

function(CUDA_move_toolkit_include_dirs_to_end)
  get_property(tmp_include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
  CUDA_remove_toolkit_include_dirs(tmp_include_dirs)
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES ${tmp_include_dirs})
  include_directories(SYSTEM "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
  get_property(tmp_include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
  message("\nInclude directories = ${tmp_include_dirs}\n")
endfunction()
