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


# List of currently used arch values
set(CUDA_known_archs "35" "50" "52" "60" "61" "70" "75")

set(CUDA_TARGET_ARCHS ${CUDA_known_archs} CACHE STRING "List of target CUDA architectures")

# Find if passing `flags` to nvcc producess success or failure
# Unix only
#
# Equivalent to dry-running preprocessing on /dev/null as .cu file
# and checking the exit code
# $ nvcc ${flags} --dryrun -E -x cu /dev/null
#
# @param out_status   TRUE iff exit code is 0, FALSE otherwise
# @param nvcc_bin     nvcc binary to use in shell invocation
# @param flags        flags to check
# @return out_status
function(CUDA_check_nvcc_flag out_status nvcc_bin flags)
  set(preprocess_empty_cu_file "--dryrun" "-E" "-x" "cu" "/dev/null")
  set(nvcc_command ${flags} ${preprocess_empty_cu_file})
  # Run nvcc and check the exit status
  execute_process(COMMAND ${nvcc_bin} ${nvcc_command}
                  RESULT_VARIABLE tmp_out_status
                  OUTPUT_QUIET
                  ERROR_QUIET)
  if (${tmp_out_status} EQUAL 0)
    set(${out_status} TRUE PARENT_SCOPE)
  else()
    set(${out_status} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Given the list of arch values, check which are supported by
# nvcc found in CUDA_TOOLKIT_ROOT_DIR. Requires CUDA to be set up in CMake.
#
# @param out_arch_values_allowed  List of arch values supported by nvcc
# @param arch_values_to_check     List of values to be checked against nvcc
#                                 for example: 60;61;70;75
# @return out_arch_values_allowed
function(CUDA_find_supported_arch_values out_arch_values_allowed arch_values_to_check)
  if (NOT CUDA_FOUND)
    message(ERROR "CUDA is needed to check supported architecture values")
  endif()
  # allow the user to pass the list like a normal variable
  set(arch_list ${arch_values_to_check} ${ARGN})
  set(nvcc "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
  foreach(arch IN LISTS arch_list ITEMS)
    CUDA_check_nvcc_flag(supported ${nvcc} "-arch=sm_${arch}")
    if (supported)
      set(out_list ${out_list} ${arch})
    endif()
  endforeach(arch)
  set(${out_arch_values_allowed} ${out_list} PARENT_SCOPE)
endfunction()

# Generate -gencode arch=compute_XX,code=sm_XX for list of supported arch values
# List should be sorted in increasing order.
# The last arch value will be repeated as -gencode arch=compute_XX,code=compute_XX
# to ensure the generation of PTX for most recent virtual architecture
# and maintain forward compatibility
#
# @param out_args_string  output string containing appropriate CUDA_NVCC_FLAGS
# @param arch_values      list of arch values to use
# @return out_args_string
function(CUDA_get_gencode_args out_args_string arch_values)
  # allow the user to pass the list like a normal variable
  set(arch_list ${arch_values} ${ARGN})
  set(out "")
  foreach(arch IN LISTS arch_list)
    set(out "${out} -gencode arch=compute_${arch},code=sm_${arch}")
  endforeach(arch)
  # Repeat the last one as to ensure the generation of PTX for most
  # recent virtual architecture for forward compatibility
  list(GET arch_list -1 last_arch)
  set(out "${out} -gencode arch=compute_${last_arch},code=compute_${last_arch}")
  set(${out_args_string} ${out} PARENT_SCOPE)
endfunction()
