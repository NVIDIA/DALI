# Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# caffe_parse_header
# Reads set of version defines from the header file
# Usage:
#   caffe_parse_header(<file> <define1> <define2> <define3> ..)
#
# From https://github.com/caffe2/caffe2/blob/d42952c5d7f9192b919cce1b7f7c00e4d825625f/cmake/Utils.cmake
# Which derives from https://github.com/BVLC/caffe/blob/master/cmake/Utils.cmake
#
# All contributions by the University of California:
# Copyright (c) 2014-2017 The Regents of the University of California (Regents)
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2014-2017, the respective contributors
# All rights reserved.
#
# Caffe uses a shared copyright model: each contributor holds copyright over
# their contributions to Caffe. The project versioning records all such
# contribution and copyright details. If a contributor wants to further mark
# their specific copyright on a particular contribution, they should indicate
# their copyright solely in the commit message of the change when it is
# committed.
#
# LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
if (POLICY CMP0175)
  cmake_policy(SET CMP0175 NEW)
endif()

macro(caffe_parse_header FILENAME FILE_VAR)
  set(vars_regex "")
  set(__parnet_scope OFF)
  set(__add_cache OFF)
  foreach(name ${ARGN})
    if("${name}" STREQUAL "PARENT_SCOPE")
      set(__parnet_scope ON)
    elseif("${name}" STREQUAL "CACHE")
      set(__add_cache ON)
    elseif(vars_regex)
      set(vars_regex "${vars_regex}|${name}")
    else()
      set(vars_regex "${name}")
    endif()
  endforeach()
  if(EXISTS "${FILENAME}")
    file(STRINGS "${FILENAME}" ${FILE_VAR} REGEX "#define[ \t]+(${vars_regex})[ \t]+[0-9]+" )
  else()
    unset(${FILE_VAR})
  endif()
  foreach(name ${ARGN})
    if(NOT "${name}" STREQUAL "PARENT_SCOPE" AND NOT "${name}" STREQUAL "CACHE")
      if(${FILE_VAR})
        if(${FILE_VAR} MATCHES ".+[ \t]${name}[ \t]+([0-9]+).*")
          string(REGEX REPLACE ".+[ \t]${name}[ \t]+([0-9]+).*" "\\1" ${name} "${${FILE_VAR}}")
        else()
          set(${name} "")
        endif()
        if(__add_cache)
          set(${name} ${${name}} CACHE INTERNAL "${name} parsed from ${FILENAME}" FORCE)
        elseif(__parnet_scope)
          set(${name} "${${name}}" PARENT_SCOPE)
        endif()
      else()
        unset(${name} CACHE)
      endif()
    endif()
  endforeach()
endmacro()

# check_and_add_cmake_submodule
# Checks for presence of a git submodule that includes a CMakeLists.txt
# Usage:
#   check_and_add_cmake_submodule(<submodule_path> ..)
macro(check_and_add_cmake_submodule SUBMODULE_PATH)
  if(NOT EXISTS ${SUBMODULE_PATH}/CMakeLists.txt)
    message(FATAL_ERROR "File ${SUBMODULE_PATH}/CMakeLists.txt not found. "
                        "Did you forget to `git clone --recursive`? Try this:\n"
                        "  cd ${PROJECT_SOURCE_DIR} && \\\n"
                        "  git submodule sync --recursive && \\\n"
                        "  git submodule update --init --recursive && \\\n"
                        "  cd -\n")
  endif()
  add_subdirectory(${SUBMODULE_PATH} ${ARGN})
endmacro(check_and_add_cmake_submodule)

# Helper function to remove elements from a variable
function (remove TARGET INPUT)
  foreach(ITEM ${ARGN})
    list(REMOVE_ITEM INPUT "${ITEM}")
  endforeach()
  set(${TARGET} ${INPUT} PARENT_SCOPE)
endfunction(remove)

macro(get_dali_version FILENAME FILE_VAR)
  if(EXISTS "${FILENAME}")
    file(STRINGS "${FILENAME}" ${FILE_VAR} REGEX "[0-9]+\.[0-9]+\.[0-9]+" )
  else()
    set(${FILE_VAR} "0.0.0")
  endif()
  message("-- DALI version: " ${${FILE_VAR}})
endmacro()

macro(get_dali_extra_version FILENAME VERSION_VAR)
  if(EXISTS "${FILENAME}")
    file(STRINGS "${FILENAME}" ${VERSION_VAR} LIMIT_INPUT 40)
  else()
    set(${VERSION_VAR} "0000000000000000000000000000000000000000")
  endif()
  message("-- DALI_extra version: " ${${VERSION_VAR}})
endmacro()


# add a post-build step to the provided target which copies
# files or directories recursively from SRC to DST
# create phony target first (if not exists with given name yet)
# and add comand attached to it
macro(copy_post_build TARGET_NAME SRC DST)
    if (NOT (TARGET copy_post_build_target))
        add_custom_target(copy_post_build_target ALL)
    endif()
    if (NOT (TARGET install_${TARGET_NAME}))
        add_custom_target(install_${TARGET_NAME} ALL
             DEPENDS ${TARGET_NAME}
        )
        add_dependencies(copy_post_build_target install_${TARGET_NAME})
    endif()

    add_custom_command(
    TARGET install_${TARGET_NAME}
    POST_BUILD
    COMMAND mkdir -p "${DST}" && cp -r
            "${SRC}"
            "${DST}")
endmacro(copy_post_build)


# Glob for source files in current dir
# Usage: collect_sources(<SRCS_VAR_NAME> [PARENT_SCOPE])
#
# Adds source files globed in current dir to variable SRCS_VAR_NAME
# at the current scope, and optionally at PARENT_SCOPE.
#
macro(collect_sources DALI_SRCS_GROUP)
  cmake_parse_arguments(
    COLLECT_SOURCES # prefix of output variables
    "PARENT_SCOPE" # all options for the respective macro
    "" # one value keywords
    "" # multi value keywords
    ${ARGV})

  file(GLOB collect_sources_tmp *.cc *.cu *.c)
  file(GLOB collect_sources_tmp_test *_test.cc *_test.cu *_test.c)
  remove(collect_sources_tmp "${collect_sources_tmp}" ${collect_sources_tmp_test})
  set(${DALI_SRCS_GROUP} ${${DALI_SRCS_GROUP}} ${collect_sources_tmp})
  if (COLLECT_SOURCES_PARENT_SCOPE)
    set(${DALI_SRCS_GROUP} ${${DALI_SRCS_GROUP}} PARENT_SCOPE)
  endif()
endmacro(collect_sources)

# Glob for test source files in current dir
# Usage: collect_test_sources(<SRCS_VAR_NAME> [PARENT_SCOPE])
#
# Adds test source files globed in current dir to variable SRCS_VAR_NAME
# at the current scope, and optionally at PARENT_SCOPE.
#
macro(collect_test_sources DALI_TEST_SRCS_GROUP)
  cmake_parse_arguments(
    COLLECT_TEST_SOURCES # prefix of output variables
    "PARENT_SCOPE" # all options for the respective macro
    "" # one value keywords
    "" # multi value keywords
    ${ARGV})

  file(GLOB collect_test_sources_tmp_test *_test.cc *_test.cu *_test.c)
  set(${DALI_TEST_SRCS_GROUP} ${${DALI_TEST_SRCS_GROUP}} ${collect_test_sources_tmp_test})
  if (COLLECT_TEST_SOURCES_PARENT_SCOPE)
    set(${DALI_TEST_SRCS_GROUP} ${${DALI_TEST_SRCS_GROUP}} PARENT_SCOPE)
  endif()
endmacro()


# Glob for the header files in current dir
# Usage: collect_headers(<HDRS_VAR_NAME> [PARENT_SCOPE] [INCLUDE_TEST])
#
# Adds *.h files to HDRS_VAR_NAME list at the current scope and optionally at PARENT_SCOPE.
# Does not collect files that contain `test` substring it the filename,
# unless INCLUDE_TEST is specified.
#
macro(collect_headers DALI_HEADERS_GROUP)
cmake_parse_arguments(
  COLLECT_HEADERS # prefix of output variables
  "PARENT_SCOPE;INCLUDE_TEST" # all options for the respective macro
  "" # one value keywords
  "" # multi value keywords
  ${ARGV})

  file(GLOB collect_headers_tmp *.h *.hpp *.cuh *.inl)
  set(${DALI_HEADERS_GROUP} ${${DALI_HEADERS_GROUP}} ${collect_headers_tmp})
  # We remove filenames containing substring test
  if(NOT COLLECT_HEADERS_INCLUDE_TEST)
    file(GLOB collect_headers_tmp *test*)
    remove(${DALI_HEADERS_GROUP} "${${DALI_HEADERS_GROUP}}" ${collect_headers_tmp})
  endif()
  if(COLLECT_HEADERS_PARENT_SCOPE)
    set(${DALI_HEADERS_GROUP} ${${DALI_HEADERS_GROUP}} PARENT_SCOPE)
  endif()
endmacro(collect_headers)

# Add a define for build option.
# for option(BUILD_FAUTRE "feature description") creates a FAUTRE_ENABLED definition
# passed to compiler, with appropriate value based on the value of the option.
#
function(propagate_option BUILD_OPTION_NAME)
  string(REPLACE "BUILD_" "" OPTION_NAME ${BUILD_OPTION_NAME})
  set(DEFINE_NAME ${OPTION_NAME}_ENABLED)
  if (${BUILD_OPTION_NAME})
    message(STATUS "${BUILD_OPTION_NAME} -- ON")
    add_definitions(-D${DEFINE_NAME}=1)
  else()
    message(STATUS "${BUILD_OPTION_NAME} -- OFF")
    add_definitions(-D${DEFINE_NAME}=0)
  endif()
endfunction(propagate_option)

function(add_sources_to_lint LINT_TARGET LINT_EXTRA LIST_SRC)
  add_custom_command(
    TARGET ${LINT_TARGET}
    POST_BUILD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND ${LINT_COMMAND} ${LINT_EXTRA} ${LIST_SRC}
    )
endfunction(add_sources_to_lint)

function(add_check_gtest_target TARGET_NAME BINARY BINARY_DIR)
  add_custom_target(${TARGET_NAME})
  add_custom_command(
    TARGET ${TARGET_NAME}
    POST_BUILD
    WORKING_DIRECTORY ${BINARY_DIR}
    COMMAND ${BINARY}
  )
  add_dependencies("check-gtest" ${TARGET_NAME})
endfunction(add_check_gtest_target)

function(parse_cuda_version CUDA_VERSION CUDA_VERSION_MAJOR_VAR CUDA_VERSION_MINOR_VAR CUDA_VERSION_PATCH_VAR CUDA_VERSION_SHORT_VAR CUDA_VERSION_SHORT_DIGIT_ONLY_VAR)
  string(REPLACE "." ";" CUDA_VERSION_LIST ${CUDA_VERSION})
  list(GET CUDA_VERSION_LIST 0 ${CUDA_VERSION_MAJOR_VAR})
  list(GET CUDA_VERSION_LIST 1 ${CUDA_VERSION_MINOR_VAR})
  string(REGEX MATCH "^[0-9]*$" ${CUDA_VERSION_MAJOR_VAR}  ${${CUDA_VERSION_MAJOR_VAR}})
  string(REGEX MATCH "^[0-9]*$" ${CUDA_VERSION_MINOR_VAR}  ${${CUDA_VERSION_MINOR_VAR}})

  list(LENGTH CUDA_VERSION_LIST LIST_LENGTH)
  if (${LIST_LENGTH} GREATER 2)
    list(GET CUDA_VERSION_LIST 2 ${CUDA_VERSION_PATCH_VAR})
    string(REGEX MATCH "^[0-9]*$" ${CUDA_VERSION_PATCH_VAR}  ${${CUDA_VERSION_PATCH_VAR}} PARENT_SCOPE)
  endif()

  if ("${${CUDA_VERSION_MAJOR_VAR}}" STREQUAL "" OR "${${CUDA_VERSION_MINOR_VAR}}" STREQUAL "")
    message(FATAL_ERROR "CUDA version is not valid: ${CUDA_VERSION}")
  endif()

  set(${CUDA_VERSION_MAJOR_VAR} "${${CUDA_VERSION_MAJOR_VAR}}" PARENT_SCOPE)
  set(${CUDA_VERSION_MINOR_VAR} "${${CUDA_VERSION_MINOR_VAR}}" PARENT_SCOPE)
  set(${CUDA_VERSION_PATCH_VAR} "${${CUDA_VERSION_PATCH_VAR}}" PARENT_SCOPE)
  message(STATUS "CUDA version: ${CUDA_VERSION}, major: ${${CUDA_VERSION_MAJOR_VAR}}, minor: ${${CUDA_VERSION_MINOR_VAR}}, patch: ${${CUDA_VERSION_PATCH_VAR}}, short: ${${CUDA_VERSION_SHORT_VAR}}, digit-only: ${${CUDA_VERSION_SHORT_DIGIT_ONLY_VAR}}")

  # when building for any version >= 11.0 use CUDA compatibility mode and claim it is a CUDA 110 package
  # TF build image uses cmake 3.5 with not GREATER_EQUAL so split it to GREATER OR EQUAL
  if ((${${CUDA_VERSION_MAJOR_VAR}} GREATER "11" OR ${${CUDA_VERSION_MAJOR_VAR}} EQUAL "11") AND ${${CUDA_VERSION_MINOR_VAR}} GREATER "0")
     set(${CUDA_VERSION_MINOR_VAR} "0")
     set(${CUDA_VERSION_PATCH_VAR} "0")
     set(${CUDA_VERSION_MINOR_VAR} "0" PARENT_SCOPE)
     set(${CUDA_VERSION_PATCH_VAR} "0" PARENT_SCOPE)
  endif()
  set(${CUDA_VERSION_SHORT_VAR} "${${CUDA_VERSION_MAJOR_VAR}}.${${CUDA_VERSION_MINOR_VAR}}" PARENT_SCOPE)
  set(${CUDA_VERSION_SHORT_DIGIT_ONLY_VAR} "${${CUDA_VERSION_MAJOR_VAR}}${${CUDA_VERSION_MINOR_VAR}}" PARENT_SCOPE)

  message(STATUS "Compatible CUDA version: major: ${${CUDA_VERSION_MAJOR_VAR}}, minor: ${${CUDA_VERSION_MINOR_VAR}}, patch: ${${CUDA_VERSION_PATCH_VAR}}, short: ${${CUDA_VERSION_SHORT_VAR}}, digit-only: ${${CUDA_VERSION_SHORT_DIGIT_ONLY_VAR}}")
endfunction()

# Build a .so library variant for each python version provided in PYTHON_VERSIONS variable
# if it is accesible during the build time. The library sufix is provided and specific for each python version
#
# supported options:
# <TARGET_NAME> - umbrella target name used for this set of libraries; two additional targets are created,
#              which can be used to set PUBLIC and PRIVATE target properties, respectively:
#              TARGET_NAME_public
#              TARGET_NAME_private.
#              Properties for these targets, such as link dependencies, includes or compilation flags,
#              must be set with INTERFACE keyword, but they are propagated as PUBLIC/PRIVATE
#              properties to all version-specific libraries.
# OUTPUT_NAME - library name used for this build
# PREFIX - library prefix, if none is provided, the library will be named ${TARGET_NAME}.python_specific_extension
# OUTPUT_DIR - ouptut directory of the build library
# PUBLIC_LIBS - list of libraries that should be linked in as a public one
# PRIV_LIBS - list of libraries that should be linked in as a private one
# SRC - list of source code files
function(build_per_python_lib)
    set(oneValueArgs TARGET_NAME OUTPUT_NAME OUTPUT_DIR PREFIX)
    set(multiValueArgs PRIV_LIBS PUBLIC_LIBS SRC EXCLUDE_LIBS)

    cmake_parse_arguments(PARSE_ARGV 1 PYTHON_LIB_ARG "${options}" "${oneValueArgs}" "${multiValueArgs}")

    set(PYTHON_LIB_ARG_TARGET_NAME ${ARGV0})
    add_custom_target(${PYTHON_LIB_ARG_TARGET_NAME} ALL)

    # global per target interface library, common for all python variants
    add_library(${PYTHON_LIB_ARG_TARGET_NAME}_public INTERFACE)
    add_library(${PYTHON_LIB_ARG_TARGET_NAME}_private INTERFACE)

    target_sources(${PYTHON_LIB_ARG_TARGET_NAME}_private INTERFACE ${PYTHON_LIB_ARG_SRC})

    target_link_libraries(${PYTHON_LIB_ARG_TARGET_NAME}_public INTERFACE ${PYTHON_LIB_ARG_PUBLIC_LIBS})
    target_link_libraries(${PYTHON_LIB_ARG_TARGET_NAME}_private INTERFACE ${PYTHON_LIB_ARG_PRIV_LIBS})
    target_link_libraries(${PYTHON_LIB_ARG_TARGET_NAME}_private INTERFACE "-Wl,--exclude-libs,${PYTHON_LIB_ARG_EXCLUDE_LIBS}")

    target_include_directories(${PYTHON_LIB_ARG_TARGET_NAME}_private
                               INTERFACE "${PYBIND11_INCLUDE_DIR}"
                               INTERFACE "${pybind11_INCLUDE_DIR}")

    foreach(PYVER ${PYTHON_VERSIONS})

        set(PYTHON_LIB_TARGET_FOR_PYVER "${PYTHON_LIB_ARG_TARGET_NAME}_${PYVER}")
        # check if listed python versions are accesible
        execute_process(
            COMMAND python${PYVER}-config --help
            RESULT_VARIABLE PYTHON_EXISTS OUTPUT_QUIET)

        if (${PYTHON_EXISTS} EQUAL 0)
            execute_process(
                COMMAND python${PYVER}-config --extension-suffix
                OUTPUT_VARIABLE PYTHON_SUFIX)
            # remove newline and the extension
            string(REPLACE ".so\n" "" PYTHON_SUFIX "${PYTHON_SUFIX}")

            execute_process(
                COMMAND python${PYVER}-config --includes
                OUTPUT_VARIABLE PYTHON_INCLUDES)
            # split and make it a list
            string(REPLACE "-I" "" PYTHON_INCLUDES "${PYTHON_INCLUDES}")
            string(REPLACE "\n" "" PYTHON_INCLUDES "${PYTHON_INCLUDES}")
            separate_arguments(PYTHON_INCLUDES)

            add_library(${PYTHON_LIB_TARGET_FOR_PYVER} SHARED)

            set_target_properties(${PYTHON_LIB_TARGET_FOR_PYVER}
                                    PROPERTIES
                                    LIBRARY_OUTPUT_DIRECTORY ${PYTHON_LIB_ARG_OUTPUT_DIR}
                                    PREFIX "${PYTHON_LIB_ARG_PREFIX}"
                                    OUTPUT_NAME ${PYTHON_LIB_ARG_OUTPUT_NAME}${PYTHON_SUFIX})
            # add includes
            foreach(incl_dir ${PYTHON_INCLUDES})
                target_include_directories(${PYTHON_LIB_TARGET_FOR_PYVER} PRIVATE ${incl_dir})
            endforeach(incl_dir)

            # add interface dummy lib as a dependnecy to easilly propagate options we could set from the above
            target_link_libraries(${PYTHON_LIB_TARGET_FOR_PYVER} PUBLIC ${PYTHON_LIB_ARG_TARGET_NAME}_public)
            target_link_libraries(${PYTHON_LIB_TARGET_FOR_PYVER} PRIVATE ${PYTHON_LIB_ARG_TARGET_NAME}_private)

            add_dependencies(${PYTHON_LIB_ARG_TARGET_NAME} ${PYTHON_LIB_TARGET_FOR_PYVER})
        endif()

    endforeach(PYVER)

endfunction()

# get default compiler include paths, needed by the stub generator
# starting from 3.14.0 CMake will have that inside CMAKE_${LANG}_IMPLICIT_INCLUDE_DIRECTORIES
macro(DETERMINE_GCC_SYSTEM_INCLUDE_DIRS _lang _compiler _flags _result)
    file(WRITE "${CMAKE_BINARY_DIR}/CMakeFiles/dummy" "\n")
    separate_arguments(_buildFlags UNIX_COMMAND "${_flags}")
    execute_process(COMMAND ${_compiler} ${_buildFlags} -v -E -x ${_lang} -dD dummy
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/CMakeFiles OUTPUT_QUIET
                    ERROR_VARIABLE _gccOutput)
    file(REMOVE "${CMAKE_BINARY_DIR}/CMakeFiles/dummy")
    if ("${_gccOutput}" MATCHES "> search starts here[^\n]+\n *(.+) *\n *End of (search) list")
        set(${_result} ${CMAKE_MATCH_1})
        string(REPLACE "\n" " " ${_result} "${${_result}}")
        separate_arguments(${_result})
    endif ()
endmacro()

function(custom_filter CURR_DIR ORIG_FILE_LIST EXTRA_FILE_LIST INCLUDE_PATTERNS EXCLUDE_PATTERNS)
  if (NOT ${INCLUDE_PATTERNS} STREQUAL "")
    file(GLOB_RECURSE ${ORIG_FILE_LIST} RELATIVE ${${CURR_DIR}}
        "${${INCLUDE_PATTERNS}}")

    foreach(file IN LISTS ${EXTRA_FILE_LIST})
      list(APPEND ${ORIG_FILE_LIST} "${file}")
    endforeach()

    foreach(exclude_pattern IN LISTS ${EXCLUDE_PATTERNS})
      file(GLOB_RECURSE excluded RELATIVE ${${CURR_DIR}} "${exclude_pattern}")
      remove(${ORIG_FILE_LIST} "${${ORIG_FILE_LIST}}" ${excluded})
    endforeach()

    message("Filtering ${ORIG_FILE_LIST}")
    message("Including ${INCLUDE_PATTERNS}: ${${INCLUDE_PATTERNS}}")
    message("Excluding ${EXCLUDE_PATTERNS}: ${${EXCLUDE_PATTERNS}}")
    message("${${ORIG_FILE_LIST}}")
    set(${ORIG_FILE_LIST} ${${ORIG_FILE_LIST}} PARENT_SCOPE)
  endif()

endfunction()

function(get_link_libraries OUTPUT_LIST TARGET)
    get_target_property(IMPORTED ${TARGET} IMPORTED)
    list(APPEND VISITED_TARGETS ${TARGET})
    if (IMPORTED)
        get_target_property(LIBS ${TARGET} INTERFACE_LINK_LIBRARIES)
    else()
        get_target_property(LIBS ${TARGET} LINK_LIBRARIES)
    endif()
    set(LIB_FILES "")
    foreach(LIB ${LIBS})
        if (TARGET ${LIB})
            list(FIND VISITED_TARGETS ${LIB} VISITED)
            if (${VISITED} EQUAL -1)
                get_target_property(type ${LIB} TYPE)
                if (NOT ${type} STREQUAL "INTERFACE_LIBRARY")
                    get_target_property(LIB_FILE ${LIB} LOCATION)
                    get_filename_component(LIB_FILE ${LIB_FILE} NAME)
                endif()
                get_link_libraries(LINK_LIB_FILES ${LIB})
                list(APPEND LIB_FILES ${LIB_FILE} ${LINK_LIB_FILES})
            endif()
        endif()
    endforeach()
    set(VISITED_TARGETS ${VISITED_TARGETS} PARENT_SCOPE)
    set(${OUTPUT_LIST} ${LIB_FILES} PARENT_SCOPE)
endfunction()
