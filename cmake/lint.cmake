# Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
# Get the project dir - not set for this target
get_filename_component(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(LINT_COMMAND python ${PROJECT_SOURCE_DIR}/third_party/cpplint.py)
set(DALI_SRC_DIR "${PROJECT_SOURCE_DIR}/dali")
set(DALI_INC_DIR "${PROJECT_SOURCE_DIR}/include")
file(GLOB_RECURSE LINT_SRC "${DALI_SRC_DIR}/*.cc" "${DALI_SRC_DIR}/*.h" "${DALI_SRC_DIR}/*.cu" "${DALI_SRC_DIR}/*.cuh")
file(GLOB_RECURSE LINT_INC "${DALI_INC_DIR}/*.h" "${DALI_INC_DIR}/*.cuh" "${DALI_INC_DIR}/*.inc" "${DALI_SRC_DIR}/*.inl")

# Excluded files

# nvdecoder
list(REMOVE_ITEM LINT_SRC
    ${DALI_SRC_DIR}/core/dynlink_cuda.cc
    ${DALI_SRC_DIR}/pipeline/operators/reader/nvdecoder/dynlink_nvcuvid.h
    ${DALI_SRC_DIR}/pipeline/operators/reader/nvdecoder/dynlink_cuviddec.h
    ${DALI_SRC_DIR}/pipeline/operators/reader/nvdecoder/dynlink_nvcuvid.cc
)

list(REMOVE_ITEM LINT_INC
  ${DALI_INC_DIR}/core/dynlink_cuda.h
)

# cuTT
list(REMOVE_ITEM LINT_SRC
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cutt.h
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cutt.cc
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cuttplan.h
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cuttplan.cc
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cuttkernel.cu
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cuttkernel.h
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/calls.h
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cuttGpuModel.h
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cuttGpuModel.cc
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cuttGpuModelKernel.h
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cuttGpuModelKernel.cu
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/CudaMemcpy.h
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/CudaMemcpy.cu
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/CudaUtils.h
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/CudaUtils.cu
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/cuttTypes.h
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/int_vector.h
    ${DALI_SRC_DIR}/pipeline/operators/transpose/cutt/LRUCache.h
)

execute_process(
  COMMAND ${LINT_COMMAND} --linelength=100 --root=include ${LINT_INC}
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  RESULT_VARIABLE LINT_RESULT
  ERROR_VARIABLE LINT_ERROR
  OUTPUT_QUIET
)

execute_process(
  COMMAND ${LINT_COMMAND} --linelength=100 ${LINT_SRC}
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  RESULT_VARIABLE LINT_RESULT
  ERROR_VARIABLE LINT_ERROR
  OUTPUT_QUIET
)

if(LINT_RESULT)
    message(FATAL_ERROR "Lint failed: ${LINT_ERROR}")
else()
    message(STATUS "Lint OK")
endif()
