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
    ${DALI_SRC_DIR}/operators/reader/nvdecoder/nvcuvid.h
    ${DALI_SRC_DIR}/operators/reader/nvdecoder/cuviddec.h
    ${DALI_SRC_DIR}/operators/reader/nvdecoder/dynlink_nvcuvid.cc
    ${DALI_SRC_DIR}/operators/reader/nvdecoder/dynlink_nvcuvid.h
)

list(REMOVE_ITEM LINT_INC
  ${DALI_INC_DIR}/dali/core/dynlink_cuda.h
)

# cuTT
list(REMOVE_ITEM LINT_SRC
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cutt.h
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cutt.cc
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cuttplan.h
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cuttplan.cc
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cuttkernel.cu
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cuttkernel.h
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/calls.h
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cuttGpuModel.h
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cuttGpuModel.cc
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cuttGpuModelKernel.h
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cuttGpuModelKernel.cu
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/CudaMemcpy.h
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/CudaMemcpy.cu
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/CudaUtils.h
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/CudaUtils.cu
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/cuttTypes.h
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/int_vector.h
    ${DALI_SRC_DIR}/operators/generic/transpose/cutt/LRUCache.h
)

set(LINT_TARGET lint)

add_custom_target(${LINT_TARGET})
add_sources_to_lint(${LINT_TARGET} "--quiet;--linelength=100;--root=${PROJECT_SOURCE_DIR}" "${LINT_SRC}")
add_sources_to_lint(${LINT_TARGET} "--quiet;--linelength=100;--root=${PROJECT_SOURCE_DIR}/include" "${LINT_INC}")
