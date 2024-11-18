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
  daliSuccess = 0,
  daliNotReady,
  daliErrorInvalidHandle,
  daliErrorInvalidArgument,
  daliErrorInvalidType,
  daliErrorInvalidOperation,
  daliErrorOutOfRange,

  daliErrorFileNotFound,
  daliErrorIOError,

  daliErrorInternal,
  daliErrorUnloading,

  daliErrorOutOfMemory = 0x100,
  daliErrorCUDAError = 0x1000,

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

typedef _DALIPipelineParams {

  int prefetch_queue_depth;
} daliPipelineParams_t;

daliError_t daliPipelineCreate(daliPipeline_h *out_pipe_handle, const daliPipelineParams_t *params);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DALI_DALI_H_
