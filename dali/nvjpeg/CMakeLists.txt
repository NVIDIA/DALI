# Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Generate necessary stub for nvjpeg if needed

DETERMINE_GCC_SYSTEM_INCLUDE_DIRS("c++" "${CMAKE_CXX_COMPILER}" "${CMAKE_CXX_FLAGS}" INFERED_COMPILER_INCLUDE)

# transform a list of paths into a list of include directives
set(DEFAULT_COMPILER_INCLUDE)
foreach(incl_dir ${INFERED_COMPILER_INCLUDE})
  set(DEFAULT_COMPILER_INCLUDE "${DEFAULT_COMPILER_INCLUDE} -I${incl_dir}")
endforeach(incl_dir)
separate_arguments(DEFAULT_COMPILER_INCLUDE UNIX_COMMAND  "${DEFAULT_COMPILER_INCLUDE}")

if(WITH_DYNAMIC_NVJPEG)
  set(NVJPEG_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvjpeg_gen.cc")
  add_custom_command(
    OUTPUT ${NVJPEG_GENERATED_STUB}
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py --unique_prefix=Nvjpeg --
    "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/nvjpeg.json" ${NVJPEG_GENERATED_STUB}
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/nvjpeg.h" "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"

    # for some reason QNX fails with 'too many errors emitted' is this is not set
    "-ferror-limit=0"
    ${DEFAULT_COMPILER_INCLUDE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/nvjpeg.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/nvjpeg.json"
    COMMENT "Running nvjpeg.h stub generator"
    VERBATIM)

  set_source_files_properties(${NVJPEG_GENERATED_STUB} PROPERTIES GENERATED TRUE)
  add_library(dynlink_nvjpeg STATIC nvjpeg_wrap.cc ${NVJPEG_GENERATED_STUB})
endif()
