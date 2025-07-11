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

# CUDA 13 include add a cccl subdir, that is detected by CMake and CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
# becomes a list instead of a single dir. In our code we relay on it being a single entry, so this
# code makes sure we extract the first path from the list and use it further (assuming it is the main dir)
# So we check if it is a list (; inside) and act accordingly
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

if (WITH_DYNAMIC_NPP)
    set(NPP_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_npp_gen.cc")
    add_custom_command(
        OUTPUT ${NPP_GENERATED_STUB}
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py --unique_prefix=Npp --
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/npp.json" ${NPP_GENERATED_STUB}
                    "${CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY}/npp.h"
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE}
                    # for some reason QNX fails with 'too many errors emitted' is this is not set
                    "-ferror-limit=0"
                    # let clang know which architecutre we compile for
                    "--target=${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
                    ${DEFAULT_COMPILER_INCLUDE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py
                "${CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY}/npp.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/npp.json"
        COMMENT "Running npp.h stub generator"
        VERBATIM)

    set_source_files_properties(${NPP_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_npp STATIC npp_wrap.cc ${NPP_GENERATED_STUB})
endif()
