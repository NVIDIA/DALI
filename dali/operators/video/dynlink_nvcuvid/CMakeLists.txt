# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_OPERATOR_SRCS PARENT_SCOPE)
collect_test_sources(DALI_OPERATOR_TEST_SRCS PARENT_SCOPE)

DETERMINE_GCC_SYSTEM_INCLUDE_DIRS("c++" "${CMAKE_CXX_COMPILER}" "${CMAKE_CXX_FLAGS}" INFERED_COMPILER_INCLUDE)

# transform a list of paths into a list of include directives
set(DEFAULT_COMPILER_INCLUDE)
foreach(incl_dir ${INFERED_COMPILER_INCLUDE})
  set(DEFAULT_COMPILER_INCLUDE "${DEFAULT_COMPILER_INCLUDE} -I${incl_dir}")
endforeach(incl_dir)
separate_arguments(DEFAULT_COMPILER_INCLUDE UNIX_COMMAND  "${DEFAULT_COMPILER_INCLUDE}")

string(FIND "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" ";" SEMICOLON_POS)
if(NOT "${SEMICOLON_POS}" EQUAL -1)
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE)
    foreach(incl_dir ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
       set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE} -I${incl_dir}")
    endforeach(incl_dir)
else()
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()
separate_arguments(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE UNIX_COMMAND  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE}")

set(NVCUVID_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvcuvid_gen.cc")
add_custom_command(
    OUTPUT ${NVCUVID_GENERATED_STUB}
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../../../../internal_tools/stub_generator/stub_codegen.py --unique_prefix=Nvcuvid --
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../../internal_tools/stub_generator/nvcuvid.json" ${NVCUVID_GENERATED_STUB}
                "${CMAKE_CURRENT_SOURCE_DIR}/nvcuvid.h"
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE}
                "-I${CMAKE_SOURCE_DIR}/include" "-I${CMAKE_SOURCE_DIR}"
                # for some reason QNX fails with 'too many errors emitted' is this is not set
                "-ferror-limit=0"
                # let clang know which architecutre we compile for
                "--target=${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
                ${DEFAULT_COMPILER_INCLUDE}
    DEPENDS  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../internal_tools/stub_generator/stub_codegen.py
            "${CMAKE_CURRENT_SOURCE_DIR}/nvcuvid.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/cuviddec.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../internal_tools/stub_generator/nvcuvid.json"
    COMMENT "Running nvcuvid.h stub generator"
    VERBATIM)

set_source_files_properties(${NVCUVID_GENERATED_STUB} PROPERTIES GENERATED TRUE)

# create a obj lib which need to be linked latter with the target
add_library(NVCUVID_GEN OBJECT ${NVCUVID_GENERATED_STUB})
