# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

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
    if (NOT (TARGET install_${TARGET_NAME}))
        add_custom_target(install_${TARGET_NAME} ALL
             DEPENDS ${TARGET_NAME}
        )
    endif()

    add_custom_command(
    TARGET install_${TARGET_NAME}
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

  file(GLOB collect_sources_tmp *.cc *.cu)
  file(GLOB collect_sources_tmp_test *_test.cc *_test.cu)
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

  file(GLOB collect_test_sources_tmp_test *_test.cc *_test.cu)
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

  file(GLOB collect_headers_tmp *.h *.hpp)
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
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND ${LINT_COMMAND} ${LINT_EXTRA} ${LIST_SRC}
    )
endfunction(add_sources_to_lint)

function(add_check_gtest_target TARGET_NAME BINARY BINARY_DIR)
  add_custom_target(${TARGET_NAME})
  add_custom_command(
    TARGET ${TARGET_NAME}
    WORKING_DIRECTORY ${BINARY_DIR}
    COMMAND ${BINARY}
    DEPENDS ${BINARY}
  )
  add_dependencies("check-gtest" ${TARGET_NAME})
endfunction(add_check_gtest_target)
