// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_DALI_H_
#define DALI_DALI_H_

#include <cuda_runtime_api.h>
#include <stdint.h>
#include "dali/core/api_helper.h"
#include "dali/core/dali_data_type.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DALI_API DLL_PUBLIC

typedef struct _DALIPipeline *daliPipeline_h;
typedef struct _DALIPipelineOutputs *daliPipelineOutputs_h;
typedef struct _DALITensor *daliTensor_h;
typedef struct _DALITensorList *daliTensorList_h;

typedef enum {
  DALI_STORAGE_CPU = 0,
  DALI_STORAGE_GPU = 1,
  DALI_STORAGE_FORCE_INT32 = 0x7fffffff
} daliStorageDevice_t;

/** Error codes returned by DALI functions */
typedef enum {
  DALI_SUCCESS = 0,
  DALI_NOT_READY,
  DALI_ERROR_INVALID_HANDLE,
  DALI_ERROR_INVALID_ARGUMENT,
  DALI_ERROR_INVALID_TYPE,
  DALI_ERROR_INVALID_OPERATION,
  DALI_ERROR_OUT_OF_RANGE,

  DALI_ERROR_FILE_NOT_FOUND,
  DALI_ERROR_I_O_ERROR,

  DALI_ERROR_INTERNAL,
  DALI_ERROR_UNLOADING,

  DALI_ERROR_OUT_OF_MEMORY = 0X100,
  DALI_ERROR_CUDA_ERROR = 0X1000,

  DALI_ERROR_FORCE_INT32 = 0x7fffffff
} daliError_t;


/** Returns the last error code.
 *
 * Returns the error code associate with the recent unsuccessful call in the calling thread.
 * Succesful calls do not overwrite the value.
 */
DALI_API daliError_t daliGetLastError();

/** Returns the last error message.
 *
 * Returns the detailed, context-specific message associated with the recent unsuccessful call
 * in the callling thread.
 * Succesful calls do not overwrite the value.
 * The pointer is invalidated by intervening DALI calls in the same thread.
 */
DALI_API const char *daliGetLastErrorMessage();

/** Clears the last error for the calling thread. */
DALI_API void daliClearLastError();

/** Returns a human-readable name of a given error
 *
 * The value is a pointer to a string literal. It's not invalidated other than by unloading DALI.
 */
DALI_API const char *daliGetErrorName(daliError_t error);

/** Returns a human-readable description of a given error.
 *
 * The value is a pointer to a string literal. It's not invalidated other than by unloading DALI.
 */
DALI_API const char *daliGetErrorDescription(daliError_t error);


/** Initializes DALI or increments initialization count. */
DALI_API daliError_t daliInit();

/** Decrements initialization counts and shuts down the library when the count reaches 0.
 *
 * Calling this function is optional. DALI will be shut down automatically when the program exits.
 */
DALI_API daliError_t daliShutdown();

/*** PIPELINE API ***********************************************************/

typedef enum _DALIExecType {
  /** The exeuctor processes data ahead, overlapping CPU/Mixed/GPU operators */
  DALI_EXEC_IS_PIPELINED    = 1,
  /** The executor operates in thread(s) other than the one that calls the pipeline Run */
  DALI_EXEC_IS_ASYNC        = 2,
  /** Deprecated: The executor uses separate CPU/GPU queues */
  DALI_EXEC_IS_SEPARATED    = 4,
  /** Use dynamic executor, with unrestricted operator order and aggressive memory reuse */
  DALI_EXEC_IS_DYNAMIC      = 8,

  /** Use a synchronous, non-pipelined executor; useful for debugging. */
  DALI_EXEC_SIMPLE          = 0,
  /** Use an asynchronous pipelined executor, the default one. */
  DALI_EXEC_ASYNC_PIPELINED = DALI_EXEC_IS_PIPELINED | DALI_EXEC_IS_ASYNC,
  /** Use the dynamic executor.
   *
   * The dynamic executor offers more flexibility, better memory efficiency and unrestricted
   * lifetime of the pipeline outputs at the expense of more overhead in simple pipelines. */
  DALI_EXEC_DYNAMIC         = DALI_EXEC_ASYNC_PIPELINED | DALI_EXEC_IS_DYNAMIC,
} daliExecType_t;


/** DALI Pipeline construction parameters */
typedef struct _DALIPipelineParams {
  size_t size;  /* must be sizeof(daliPipelineParams_t) */
  struct {
    uint64_t max_batch_size_present : 1;
    uint64_t num_threads_present    : 1;
    uint64_t device_id_present      : 1;
    uint64_t seed_present           : 1;
    uint64_t exec_flags_present     : 1;
    uint64_t exec_type_present      : 1;
    uint64_t enable_checkpointing_present : 1;
    uint64_t enable_memory_stats_present  : 1;
  };
  int batch_size;
  int num_threads;
  int device_id;
  int64_t seed;
  daliExecType_t exec_type;
  daliBool enable_checkpointing;
  daliBool enable_memory_stats;
} daliPipelineParams_t;

/** Describes an output of a DALI Pipeline */
typedef struct _DALIPipelineOutputDesc {
  const char *name;
  daliStorageDevice_t device;
  struct {
    unsigned dtype_present : 1;
    unsigned ndim_present : 1;
  };
  daliDataType_t dtype;
  int ndim;
} daliPipelineOutputDesc_t;

/** Creates an empty pipeline. */
DALI_API daliError_t daliPipelineCreate(
  daliPipeline_h *out_pipe_handle,
  const daliPipelineParams_t *params);

/** Creates a DALI pipeline from a serialized one.
 *
 * This function creates and deserializes a pipeline. The parameters are used to override
 * the serialized ones.
 *
 * @param out_pipe_handle [out] points to a value which will receive the handle to the newly
 *                              created pipeline
 * @param serialized_pipeline [in] a raw memory buffer containing the pipeline as protobuf
 * @param serialized_pipeline_length the length, in bytes, of the `serialized_pipeline` buffer
 * @param param_overrides [in] contains (partial) pipeline construction parameters;
 *                             the parameters specified in this structure override the corresponding
 *                             parameters deserialized from the buffer.
 */
DALI_API daliError_t daliPipelineDeserialize(
  daliPipeline_h *out_pipe_handle,
  const void *serialized_pipeline,
  size_t serialized_pipeline_size,
  const daliPipelineParams_t *param_overrides);


/** Prepares the pipeline for execution */
DALI_API daliError_t daliPipelineBuild(daliPipeline_h pipeline);

/** Runs the pipeline to fill the queues.
 *
 * DALI Pipeline can process several iterations ahead. This function pre-fills the queues.
 * If the pipeline has ExternalSource operators (or other external inputs), they need to be
 * supplied with enough data.
 * @see daliPipelineFeedInput
 * @see daliPipelineGetInputFeedCount
 */
DALI_API daliError_t daliPipelinePrefetch(daliPipeline_h pipeline);

/** Schedules one iteration.
 *
 * If the executor doesn't have DALI_EXEC_IS_ASYNC flag, the function will block until the
 * operation is complete on host.
 *
 * NOTE: The relevant device code may still be running after this function returns.
 */
DALI_API daliError_t daliPipelineRun(daliPipeline_h pipeline);

/** Pops the pipeline outputs from the pipeline's output queue.
 *
 * The outputs are ready for use on any stream.
 * When no longer used, the outputs should be freed by destroying the daliPipelineOutput object.
 */
DALI_API daliError_t daliPipelinePopOutputs(daliPipeline_h pipeline, daliPipelineOutput_h *out);

/** Gets the number of pipeline outputs */
DALI_API daliError_t daliPipelineGetOutputCount(daliPipeline_h pipeline, int *out_count);

/** Gets the number of pipeline outputs */
DALI_API daliError_t daliPipelineGetOutputDesc(
  daliPipeline_h pipeline,
  daliPipelineOutputDesc_t *out_desc,
  int index);


/** Pops the pipeline outputs from the pipeline's output queue.
 *
 * The outputs are ready for use on the provided stream.
 * When no longer used, the outputs should be freed by destroying the daliPipelineOutput object.
 *
 * This function works only with DALI_EXEC_IS_DYNAMIC.
 */
DALI_API daliError_t daliPipelinePopOutputsAsync(
  daliPipeline_h pipeline,
  daliPipelineOutputs_h *out,
  cudaStream_t stream);

/** Releases the pipeline outputs.
 *
 * This function destroys the daliPipelineOutputObject. The availability of the outputs differs
 * between different executors.
 * If DALI_EXEC_IS_DYNAMIC is used, the outputs may be used until their handles are destroyed.
 * Otherwise, the outputs must not be used after this call has been made.
 */
DALI_API daliError_t daliPipelineOutputsDestroy(daliPipelineOutputs_h out);

/** Gets index-th output.
 *
 * The handle returned by this function must be released with a call to daliTensorListDecRef
 */
DALI_API daliError_t daliPipelineOutputsGet(
  daliPipelineOutputs_h outputs,
  daliTensorList_h *out,
  int index);

/*** TensorList API **********************************************************/

DALI_API daliError_t daliTensorListCreate(
  daliTensorList_h *out,
  daliStorageDevice_t device_type,
  int device_id,
  int num_samples,
  int ndim,
  daliDataType_t dtype,
  const int **shape);

DALI_API daliError_t daliTensorListCreateFromContiguousBuffer(
  daliTensorList_h *out,
  daliStorageDevice_t device_type,
  int device_id,
  int num_samples,
  int ndim,
  daliDataType_t dtype,
  const int **shape,
  void *raw_data,
  daliRawDataDeleter_t deleter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DALI_DALI_H_
