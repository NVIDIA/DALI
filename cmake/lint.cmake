# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
get_filename_component(CMAKE_PROJECT_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(LINT_COMMAND python ${CMAKE_PROJECT_DIR}/third_party/cpplint.py)
set(DALI_SRC_DIR "${CMAKE_PROJECT_DIR}/dali")
set(DALI_INC_DIR "${CMAKE_PROJECT_DIR}/include")
file(GLOB_RECURSE LINT_SRC "${DALI_SRC_DIR}/*.cc" "${DALI_SRC_DIR}/*.h" "${DALI_SRC_DIR}/*.cu" "${DALI_SRC_DIR}/*.cuh")
file(GLOB_RECURSE LINT_INC "${DALI_INC_DIR}/*.h" "${DALI_INC_DIR}/*.cuh" "${DALI_INC_DIR}/*.inc" "${DALI_SRC_DIR}/*.inl")

# Excluded files
# rapidjson
list(REMOVE_ITEM LINT_SRC
    ${DALI_SRC_DIR}/util/rapidjson/allocators.h
    ${DALI_SRC_DIR}/util/rapidjson/cursorstreamwrapper.h
    ${DALI_SRC_DIR}/util/rapidjson/document.h
    ${DALI_SRC_DIR}/util/rapidjson/encodedstream.h
    ${DALI_SRC_DIR}/util/rapidjson/encodings.h
    ${DALI_SRC_DIR}/util/rapidjson/filereadstream.h
    ${DALI_SRC_DIR}/util/rapidjson/filewritestream.h
    ${DALI_SRC_DIR}/util/rapidjson/fwd.h
    ${DALI_SRC_DIR}/util/rapidjson/istreamwrapper.h
    ${DALI_SRC_DIR}/util/rapidjson/memorybuffer.h
    ${DALI_SRC_DIR}/util/rapidjson/memorystream.h
    ${DALI_SRC_DIR}/util/rapidjson/ostreamwrapper.h
    ${DALI_SRC_DIR}/util/rapidjson/pointer.h
    ${DALI_SRC_DIR}/util/rapidjson/prettywriter.h
    ${DALI_SRC_DIR}/util/rapidjson/rapidjson.h
    ${DALI_SRC_DIR}/util/rapidjson/reader.h
    ${DALI_SRC_DIR}/util/rapidjson/schema.h
    ${DALI_SRC_DIR}/util/rapidjson/stream.h
    ${DALI_SRC_DIR}/util/rapidjson/stringbuffer.h
    ${DALI_SRC_DIR}/util/rapidjson/writer.h
    ${DALI_SRC_DIR}/util/rapidjson/error/en.h
    ${DALI_SRC_DIR}/util/rapidjson/error/error.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/biginteger.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/diyfp.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/dtoa.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/ieee754.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/itoa.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/meta.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/pow10.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/regex.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/stack.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/strfunc.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/strtod.h
    ${DALI_SRC_DIR}/util/rapidjson/internal/swap.h
    ${DALI_SRC_DIR}/util/rapidjson/msinttypes/inttypes.h
    ${DALI_SRC_DIR}/util/rapidjson/msinttypes/stdint.h
)

# nvdecoder
list(REMOVE_ITEM LINT_SRC
    ${DALI_SRC_DIR}/util/dynlink_cuda.h
    ${DALI_SRC_DIR}/util/dynlink_cuda.cc
    ${DALI_SRC_DIR}/pipeline/operators/reader/nvdecoder/dynlink_nvcuvid.h
    ${DALI_SRC_DIR}/pipeline/operators/reader/nvdecoder/dynlink_cuviddec.h
    ${DALI_SRC_DIR}/pipeline/operators/reader/nvdecoder/dynlink_nvcuvid.cc
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
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
  RESULT_VARIABLE LINT_RESULT
  ERROR_VARIABLE LINT_ERROR
  OUTPUT_QUIET
)

execute_process(
  COMMAND ${LINT_COMMAND} --linelength=100 ${LINT_SRC}
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
  RESULT_VARIABLE LINT_RESULT
  ERROR_VARIABLE LINT_ERROR
  OUTPUT_QUIET
)

if(LINT_RESULT)
    message(FATAL_ERROR "Lint failed: ${LINT_ERROR}")
else()
    message(STATUS "Lint OK")
endif()
