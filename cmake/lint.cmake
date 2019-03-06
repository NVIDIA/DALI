# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
get_filename_component(CMAKE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(LINT_COMMAND python ${CMAKE_SOURCE_DIR}/third_party/cpplint.py)
file(GLOB_RECURSE LINT_FILES ${CMAKE_SOURCE_DIR}/dali/*.cc ${CMAKE_SOURCE_DIR}/dali/*.h ${CMAKE_SOURCE_DIR}/dali/*.cu ${CMAKE_SOURCE_DIR}/dali/*.cuh)

# Excluded files
# rapidjson
list(REMOVE_ITEM LINT_FILES
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/allocators.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/cursorstreamwrapper.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/document.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/encodedstream.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/encodings.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/filereadstream.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/filewritestream.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/fwd.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/istreamwrapper.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/memorybuffer.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/memorystream.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/ostreamwrapper.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/pointer.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/prettywriter.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/rapidjson.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/reader.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/schema.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/stream.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/stringbuffer.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/writer.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/error/en.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/error/error.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/biginteger.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/diyfp.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/dtoa.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/ieee754.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/itoa.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/meta.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/pow10.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/regex.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/stack.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/strfunc.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/strtod.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/internal/swap.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/msinttypes/inttypes.h
    ${CMAKE_SOURCE_DIR}/dali/util/rapidjson/msinttypes/stdint.h
)

# nvdecoder
list(REMOVE_ITEM LINT_FILES
    ${CMAKE_SOURCE_DIR}/dali/util/dynlink_cuda.h
    ${CMAKE_SOURCE_DIR}/dali/util/dynlink_cuda.cc
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/reader/nvdecoder/dynlink_nvcuvid.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/reader/nvdecoder/dynlink_cuviddec.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/reader/nvdecoder/dynlink_nvcuvid.cc
)

# cuTT
list(REMOVE_ITEM LINT_FILES
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cutt.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cutt.cc
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cuttplan.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cuttplan.cc
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cuttkernel.cu
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cuttkernel.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/calls.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cuttGpuModel.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cuttGpuModel.cc
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cuttGpuModelKernel.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cuttGpuModelKernel.cu
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/CudaMemcpy.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/CudaMemcpy.cu
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/CudaUtils.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/CudaUtils.cu
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/cuttTypes.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/int_vector.h
    ${CMAKE_SOURCE_DIR}/dali/pipeline/operators/transpose/cutt/LRUCache.h
)

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
