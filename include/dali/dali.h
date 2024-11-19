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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _DALIPipeline *daliPipeline_h;
typedef struct _DALITensor *daliTensor_h;
typedef struct _DALITensorList *daliTensorList_h;

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
daliError_t daliGetLastError();

/** Returns the last error message.
 *
 * Returns the detailed, context-specific message associated with the recent unsuccessful call
 * in the callling thread.
 * Succesful calls do not overwrite the value.
 * The pointer is invalidated by intervening DALI calls in the same thread.
 */
const char *daliGetLastErrorMessage();

/** Clears the last error for the calling thread. */
void daliClearLastError();

/** Returns a human-readable name of a given error
 *
 * The value is a pointer to a string literal. It's not invalidated other than by unloading DALI.
 */
const char *daliGetErrorName(daliError_t error);

/** Returns a human-readable description of a given error.
 *
 * The value is a pointer to a string literal. It's not invalidated other than by unloading DALI.
 */
const char *daliGetErrorDescription(daliError_t error);


/** Initializes DALI or increments initialization count. */
daliError_t daliInit();

/** Decrements initialization counts and shuts down the library when the count reaches 0.
 *
 * Calling this function is optional. DALI will be shut down automatically when the program exits.
 */
daliError_t daliShutdown();

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
} daliExecFlags_t;

typedef _DALIPipelineParams {
  size_t size;  /* must be sizeof(daliPipelineParams_t) */
  struct {
    uint64_t has_max_batch_size : 1;
    uint64_t has_num_threads : 1;
    uint64_t has_max_batch_size : 1;
    uint64_t has_seed : 1;
    uint64_t has_prefetch_queue_depth : 1;
    uint64_t has_enable_memory_stats : 1;
    uint64_t
  };
  int max_batch_size;
  int num_threads;

        batch_size=-1,
        num_threads=-1,
        device_id=-1,
        seed=-1,
        exec_pipelined=True,
        prefetch_queue_depth=2,
        exec_async=True,
        bytes_per_sample_hint=0,
        set_affinity=False,
        default_cuda_stream_priority=0,
        *,
        enable_memory_stats=False,
        enable_checkpointing=False,
        checkpoint=None,
        py_num_workers=1,
        py_start_method="fork",
        py_callback_pickler=None,
        output_dtype=None,
        output_ndim=None,


} daliPipelineParams_t;

daliError_t daliPipelineCreate(daliPipeline_h *out_pipe_handle, const daliPipelineParams_t *params);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DALI_DALI_H_
