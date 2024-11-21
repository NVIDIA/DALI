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

typedef struct _DALIPipeline *daliPipeline_h;
typedef struct _DALITensor *daliTensor_h;
typedef struct _DALITensorList *daliTensorList_h;

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
} daliExecType_t;

/*#define DALI_DEFINE_OPTIONAL_TYPE(value_type, ...) \
  typedef struct {                                 \
    daliBool has_value;                            \
    value_type value;                              \
  } __VA_ARGS__;


DALI_DEFINE_OPTIONAL_TYPE(daliBool, daliOptionalBool_t);
DALI_DEFINE_OPTIONAL_TYPE(int32_t, daliOptionalInt_t, daliOptionalInt32_t);
DALI_DEFINE_OPTIONAL_TYPE(uint32_t, daliOptionalInt_t, daliOptionalInt32_t);
DALI_DEFINE_OPTIONAL_TYPE(int64_t, daliOptionalInt64_t);
DALI_DEFINE_OPTIONAL_TYPE(uint64_t, daliOptionalInt64_t);
DALI_DEFINE_OPTIONAL_TYPE(float, daliOptionalFloat_t);
DALI_DEFINE_OPTIONAL_TYPE(double, daliOptionalDouble_t);*/

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

/** Creates an empty pipeline. */
daliError_t daliPipelineCreate(daliPipeline_h *out_pipe_handle, const daliPipelineParams_t *params);

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
daliError_t daliPipelineDeserialize(
  daliPipeline_h *out_pipe_handle,
  const void *serialized_pipeline,
  size_t serialized_pipeline_size,
  const daliPipelineParams_t *param_overrides);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DALI_DALI_H_
