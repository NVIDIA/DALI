# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
get_filename_component(CMAKE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(LINT_COMMAND python ${CMAKE_SOURCE_DIR}/third_party/cpplint.py)
file(GLOB_RECURSE LINT_FILES ${CMAKE_SOURCE_DIR}/dali/*.cc ${CMAKE_SOURCE_DIR}/dali/*.h ${CMAKE_SOURCE_DIR}/dali/*.cu ${CMAKE_SOURCE_DIR}/dali/*.cuh)

# Excluded files
list(REMOVE_ITEM LINT_FILES ${CMAKE_SOURCE_DIR}/dali/util/json.h)

execute_process(
  COMMAND ${LINT_COMMAND} --linelength=100 ${LINT_FILES}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  RESULT_VARIABLE LINT_RESULT
  ERROR_VARIABLE LINT_ERROR
  OUTPUT_QUIET
)

if(LINT_RESULT)
    message(FATAL_ERROR "Lint failed: ${LINT_ERROR}")
else()
    message(STATUS "Lint OK")
endif()
