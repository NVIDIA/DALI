// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef DALI_C_API_H_
#error The new DALI C API is incompatible with the old one. Please do not include both headers in one translation unit.  // NOLINT
#endif

#if (defined(__cplusplus) && __cplusplus < 201402L) || \
    (!defined(__cplusplus) && __STDC_VERSION__ < 199901L)
#error The DALI C API requires a C99 or a C++14 compiler.
#endif

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
  /** The call succeeded */
  DALI_SUCCESS = 0,
  /** The call succeeded, but didn't return a value */
  DALI_NO_DATA = 1,
  /** The call succeeded, but the queried object is not ready */
  DALI_NOT_READY = 2,

  DALI_ERROR = (int32_t)0x80000000,  // NOLINT
  /** The handle is not valid. */
  DALI_ERROR_INVALID_HANDLE,
  /** The argument is invalid. Check error message for details. */
  DALI_ERROR_INVALID_ARGUMENT,
  /** An invalid type was specified. */
  DALI_ERROR_INVALID_TYPE,
  /** A generaic user error */
  DALI_ERROR_INVALID_OPERATION,
  /** The index is out of valid range */
  DALI_ERROR_OUT_OF_RANGE,
  /** The key is not found (when getting) or is not a valid key (when setting) */
  DALI_ERROR_INVALID_KEY,

  /** An operating system routine failed. */
  DALI_ERROR_SYSTEM,
  /** A path to a file or other OS resource is invalid */
  DALI_ERROR_PATH_NOT_FOUND,
  /** An I/O operation failed */
  DALI_ERROR_IO_ERROR,
  /** An operation timed out */
  DALI_ERROR_TIMEOUT,

  /** A memory allocation failed */
  DALI_ERROR_OUT_OF_MEMORY = DALI_ERROR + 0x100,

  /** Internal error - logic error in DALI code */
  DALI_ERROR_INTERNAL = DALI_ERROR + 0x200,
  /** The library is shutting down or has shut down */
  DALI_ERROR_UNLOADING,

  /** A CUDA API call has failed */
  DALI_ERROR_CUDA_ERROR = DALI_ERROR + 0x10000,

  DALI_ERROR_FORCE_INT32 = 0x7fffffff
} daliResult_t;

/** A custom deleter
 *
 * This object aggregates a custom memory deleter, a context and a destructor.
 *
 * NOTE: This structure is typically passed by value for convenience.
 */
typedef struct _DALIDeleter {
  /** A custom user-provided context.
   *
   * If the deleter is an object, then `deleter_ctx` is its `this` pointer.
   * Stateless deleters may set it to NULL.
   */
  void *deleter_ctx;

  /** Destroys the user-provided context.
   *
   * This function is called by DALI when the deleter is no longer necessary.
   * The call is omitted if either `deleter_ctx` or `destroy_context` is NULL.
   *
   * @param deleter_ctx a custom user-provided context for the deleter
   */
  void (*destroy_context)(void *deleter_ctx);

  /** Deletes a memory buffer `data`.
   *
   * @param deleter_ctx a custom user-provided context for the deleter
   * @param data        the buffer to delete
   * @param stream      If present, the deletion must be ordered after all operations
   *                    scheduled in *stream; the deleter may either use stream-ordered deletion
   *                    or otherwise ensure that the memory is valid until all operations scheduled
   *                    on *stream prior to the call are complete.
   *                    No operations in any stream scheduled after this call may use `data`.
   */
  void (*delete_buffer)(void *deleter_ctx, void *data, const cudaStream_t *stream);
} daliDeleter_t;

/** Returns the last error code.
 *
 * Returns the error code associate with the recent unsuccessful call in the calling thread.
 * Successful calls do not overwrite the value.
 *
 * @retval Error code of the most recent unsuccessful DALI API call,
 * @retval DALI_SUCCESS  if no error has occurred in the calling thread or the error
 *                       has been cleared with a call to daliClearLastError.
 */
DALI_API daliResult_t daliGetLastError();

/** Returns the last error message.
 *
 * Returns the detailed, context-specific message associated with the recent unsuccessful call
 * in the callling thread.
 * Successful calls do not overwrite the value.
 * The pointer is invalidated by intervening DALI calls in the same thread.
 */
DALI_API const char *daliGetLastErrorMessage();

/** Clears the last error for the calling thread. */
DALI_API void daliClearLastError();

/** Returns a human-readable name of a given error
 *
 * The value is a pointer to a string literal. It's not invalidated other than by unloading DALI.
 */
DALI_API const char *daliGetErrorName(daliResult_t error);

/** Returns a human-readable description of a given error.
 *
 * The value is a pointer to a string literal. It's not invalidated other than by unloading DALI.
 */
DALI_API const char *daliGetErrorDescription(daliResult_t error);


/** Initializes DALI or increments initialization count.
 *
 * @remark If this function is not called, DALI will be initialized implicitly on the first
 *         call to DALI APIs. When using implicit initialization, `daliShutdown` should not be used.
 */
DALI_API daliResult_t daliInit();

/** Decrements initialization counts and shuts down the library when the count reaches 0.
 *
 * Calling this function is optional. DALI will be shut down automatically when the program exits.
 */
DALI_API daliResult_t daliShutdown();

/* Starts with version 2 to avoid name collision with legacy C API */
DALI_API daliResult_t daliPreallocateDeviceMemory2(size_t bytes, int device_id);

/** Allocates `bytes` bytes of device memory on device `device_id`.
 *
 * The function works by allocating and immediately freeing the specified amount of device
 * memory. This will typically release the memory back to DALI's memory pool, speeding up
 * subsequent allocations.
 */
static inline daliResult_t daliPreallocateDeviceMemory(size_t bytes, int device_id) {
  return daliPreallocateDeviceMemory2(bytes, device_id);
}

/* Starts with version 2 to avoid name collision with legacy C API */
DALI_API daliResult_t daliPreallocatePinnedMemory2(size_t bytes);

/** Allocates `bytes` bytes of device-accessible host memory.
 *
 * The function works by allocating and immediately freeing the specified amount of pinned
 * memory. This will typically release the memory back to DALI's memory pool, speeding up
 * subsequent allocations.
 */
static inline daliResult_t daliPreallocatePinnedMemory(size_t bytes) {
  return daliPreallocatePinnedMemory2(bytes);
}

/* Starts with version 2 to avoid name collision with legacy C API */
DALI_API daliResult_t daliReleaseUnusedMemory2();

/** Releases unused memory from DALI memory pools to the operating system.
 *
 * NOTE: Some of the memory pool implementations allocate memory from the OS in large chunks.
 *       If the chunk is occupied by even a tiny allocation, it will not be freed by this function.
 */
static inline daliResult_t daliReleaseUnusedMemory() {
  return daliReleaseUnusedMemory2();
}


/****************************************************************************/
/*** Pipeline API ***********************************************************/
/****************************************************************************/

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

  DALI_EXEC_FORCE_INT32 = 0x7fffffff
} daliExecType_t;

typedef enum _DALIExecFlags {
  /** If set, worker threads have CPU affinity set via NVML. */
  DALI_EXEC_FLAGS_SET_AFFINITY         = 0x00000001,

  /* TODO(michalz): Make concurrency configurable in the pipeline */
  /** Masks the part of the flags that represent the execution concurrency type. */
  DALI_EXEC_FLAGS_CONCURRENCY_MASK     = 0x0000000e,

  /** Uses the internally defined default concurrency behavior */
  DALI_EXEC_FLAGS_CONCURRENCY_DEFAULT  = 0,

  /** Operators are not executed in parallel.
   *
   * For DALI_EXEC_DYNAMIC only.
   */
  DALI_EXEC_FLAGS_CONCURRENCY_NONE     = 1 << 1,
  /** Operators with different device type (cpu, gpu, mixed) can be executed concurrently.
   * Operators with the same device are executed sequentially.
   *
   * For DALI_EXEC_DYNAMIC only.
   */
  DALI_EXEC_FLAGS_CONCURRENCY_BACKEND  = 2 << 1,
  /** Any two operators may run in parallel.
   *
   * For DALI_EXEC_DYNAMIC only.
   */
  DALI_EXEC_FLAGS_CONCURRENCY_FULL     = 3 << 1,

  /** Masks the part of the flags that represent the stream policy. */
  DALI_EXEC_FLAGS_STREAM_POLICY_MASK   = 0x00000070,

  /** Use the internally defined default stream policy.
   *
   * For DALI_EXEC_DYNAMIC only. The default policy may change with DALI version.
   */
  DALI_EXEC_FLAGS_STREAM_POLICY_DEFAULT = 0,

  /** Use a single CUDA stream for all operators that need one.
   *
   * For DALI_EXEC_DYNAMIC only.
   */
  DALI_EXEC_FLAGS_STREAM_POLICY_SINGLE = 1 << 4,

  /** Use different CUDA streams for CPU, Mixed and GPU operators.
   *
   * For DALI_EXEC_DYNAMIC only.
   */
  DALI_EXEC_FLAGS_STREAM_POLICY_PER_BACKEND = 2 << 4,

  /** Use dedicated streams for independent CUDA-enabled operators.
   *
   * For DALI_EXEC_DYNAMIC only.
   */
  DALI_EXEC_FLAGS_STREAM_POLICY_PER_OPERATOR = 3 << 4,

  DALI_EXEC_FLAGS_FORCE_INT32 = 0x7fffffff
} daliExecFlags_t;

typedef struct _DALIPrefetchQueueSizes {
  int cpu, gpu;
} daliPrefetchQueueSizes_t;

/** DALI Pipeline construction parameters */
typedef struct _DALIPipelineParams {
  struct {
    uint64_t max_batch_size_present : 1;
    uint64_t num_threads_present    : 1;
    uint64_t device_id_present      : 1;
    uint64_t seed_present           : 1;
    uint64_t exec_type_present      : 1;
    uint64_t exec_flags_present     : 1;
    uint64_t prefetch_queue_depths_present : 1;
    uint64_t enable_checkpointing_present : 1;
    uint64_t enable_memory_stats_present  : 1;
    uint64_t bytes_per_sample_hint_present : 1;
  };
  int max_batch_size;
  int num_threads;
  int device_id;
  int64_t seed;
  daliExecType_t exec_type;
  daliExecFlags_t exec_flags;
  daliPrefetchQueueSizes_t prefetch_queue_depths;
  daliBool enable_checkpointing;
  daliBool enable_memory_stats;
  size_t bytes_per_sample_hint;
} daliPipelineParams_t;

/** Describes an output of a DALI Pipeline */
typedef struct _DALIPipelineIODesc {
  const char *name;
  daliStorageDevice_t device;
  struct {
    unsigned dtype_present : 1;
    unsigned ndim_present : 1;
  };
  daliDataType_t dtype;
  int ndim;
  const char *layout;
} daliPipelineIODesc_t;

/** Creates an empty pipeline. */
DALI_API daliResult_t daliPipelineCreate(
  daliPipeline_h *out_pipe_handle,
  const daliPipelineParams_t *params);

/** Destroys a DALI pipeline. */
DALI_API daliResult_t daliPipelineDestroy(daliPipeline_h pipeline);

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
DALI_API daliResult_t daliPipelineDeserialize(
  daliPipeline_h *out_pipe_handle,
  const void *serialized_pipeline,
  size_t serialized_pipeline_size,
  const daliPipelineParams_t *param_overrides);


/** Prepares the pipeline for execution */
DALI_API daliResult_t daliPipelineBuild(daliPipeline_h pipeline);

/** Runs the pipeline to fill the queues.
 *
 * DALI Pipeline can process several iterations ahead. This function pre-fills the queues.
 * If the pipeline has ExternalSource operators (or other external inputs), they need to be
 * supplied with enough data.
 *
 * @see daliPipelineFeedInput
 * @see daliPipelineGetInputFeedCount
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_INVALID_OPERATION
 * @retval DALI_ERROR_OUT_OF_MEMORY
 *
 */
DALI_API daliResult_t daliPipelinePrefetch(daliPipeline_h pipeline);

/** Schedules one iteration.
 *
 * If the executor doesn't have DALI_EXEC_IS_ASYNC flag, the function will block until the
 * operation is complete on host.
 *
 * NOTE: The relevant device code may still be running after this function returns.
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_INVALID_OPERATION
 * @retval DALI_ERROR_OUT_OF_MEMORY
 */
DALI_API daliResult_t daliPipelineRun(daliPipeline_h pipeline);

/** Gets the required feed count for the specified input of the pipeline.
 *
 * @param pipeline        [in]  The pipeline
 * @param out_feed_count  [out] The number of batches to feed into the specified input before
 *                              `daliPipelinePrefetch` can be called.
 * @param input_name      [in]  The name of the input.
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_INVALID_KEY         if `input_name` is not a valid name of an input of the
 *                                        pipeline
 */
DALI_API daliResult_t daliPipelineGetFeedCount(
  daliPipeline_h pipeline,
  int *out_feed_count,
  const char *input_name);

typedef enum _DALIFeedInputFlags {
  /** Wait for the copy to complete. */
  DALI_FEED_INPUT_SYNC = 1,
  /** Force a copy. */
  DALI_FEED_INPUT_FORCE_COPY = 2,
  /** Do not make a copy of the input, use it directly instead. */
  DALI_FEED_INPUT_NO_COPY = 4,
  /** Masks the part of the flags that describes the copy mode. */
  DALI_FEED_INPUT_COPY_MASK = 6,

  /** GPU-only: If set, the copy is performed by a CUDA kernel instead of cudaMemcpy */
  DALI_FEED_INPUT_USE_COPY_KERNEL = 8,

  DALI_FEED_INPUT_FORCE_INT32 = 0x7fffffff
} daliFeedInputFlags_t;

/** Feeds the input `input_name` with data from `input_data`.
 *
 * @param pipeline      the pipeline
 * @param input_name    the name of the input
 * @param input_data    the tensor list containing the data
 * @param data_id       an identifier of this data batch
 * @param options       flags that modify the behavior of the function, see `daliFeedInputFlags_t`
 * @param stream        the stream on which it is safe to access the data;
 *                      if NULL, the stream associated with `input_data` is used.
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_INVALID_KEY         if `input_name` is not a valid name of an input of the
 *                                        pipeline
 */
DALI_API daliResult_t daliPipelineFeedInput(
  daliPipeline_h pipeline,
  const char *input_name,
  daliTensorList_h input_data,
  const char *data_id,
  daliFeedInputFlags_t options,
  const cudaStream_t *stream);

/** Gets the number of pipeline inputs.
 *
 * NOTE: The pipeline must be built before calling this function.
 *
 * @param pipeline        [in]  The pipeline
 * @param out_input_count [out] A pointer to the location where the number of pipeline inputs is
 *                              stored.
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_INVALID_OPERATION   the pipeline wasn't built before the call
 */
DALI_API daliResult_t daliPipelineGetInputCount(daliPipeline_h pipeline, int *out_input_count);

/** Gets a descriptor of a pipeline input specified by index.
 *
 * NOTE: The pipeline must be built before calling this function.
 *
 * @param pipeline        [in]  The pipeline
 * @param out_input_desc  [out] A pointer to the location where the descriptor is written.
 * @param index           [in]  The 0-based index of the input. See `daliPipelineGetInputCount`.
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_INVALID_OPERATION   the pipeline wasn't built before the call
 * @retval DALI_ERROR_OUT_OF_RANGE        the index is not a valid 0-based index of the an input
 */
DALI_API daliResult_t daliPipelineGetInputDescByIdx(
  daliPipeline_h pipeline,
  daliPipelineIODesc_t *out_input_desc,
  int index);

/** Gets a descriptor of a pipeline input specified by its name.
 *
 * NOTE: The pipeline must be built before calling this function.
 *
 * @param pipeline        [in]  The pipeline
 * @param out_input_desc  [out] A pointer to the location where the descriptor is written.
 * @param name            [in]  The name of the input whose descriptor to obtain.
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_INVALID_OPERATION   the pipeline wasn't built before the call
 * @retval DALI_ERROR_INVALID_KEY         if `input_name` is not a valid name of an input of the
 *                                        pipeline
 */
DALI_API daliResult_t daliPipelineGetInputDesc(
  daliPipeline_h pipeline,
  daliPipelineIODesc_t *out_input_desc,
  const char *name);

/** Gets the number of pipeline outputs.
 *
 * @param pipeline  [in]  The pipeline
 * @param out_count [out] A pointer to the location where the number of pipeline outputs is stored.
 */
DALI_API daliResult_t daliPipelineGetOutputCount(daliPipeline_h pipeline, int *out_count);

/** Gets a descriptor of the specified pipeline output.
 *
 * @param pipeline  [in]  The pipeline
 * @param out_desc  [out] A pointer to the location where the descriptor is written.
 * @param index     [in]  The 0-based index of the output. See `daliPipelineGetOutputCount`.
 *
 * NOTE: The names returned by this function match those specified when defining the pipeline,
 *       but don't necessarily indicate the output operators. When building the pipeline,
 *       operators may be added (e.g. to guarantee dense storage of the outputs) or removed
 *       (in the process of graph optimization).
 */
DALI_API daliResult_t daliPipelineGetOutputDesc(
  daliPipeline_h pipeline,
  daliPipelineIODesc_t *out_desc,
  int index);

/** Pops the pipeline outputs from the pipeline's output queue.
 *
 * The outputs are ready for use on any stream.
 * When no longer used, the outputs must be freed by destroying the `daliPipelineOutput` object.
 *
 * @param pipeline [in]  The pipeline whose outputs are to be obtained
 * @param out      [out] A pointer to the output handle. The handle is NULL if the function
 *                       reports an error.
 *
 * @return This function will report errors that occurred asynchronously when preparing the
 *         relevant data batch. If an error is reported, the output handle is NULL.
 *
 */
DALI_API daliResult_t daliPipelinePopOutputs(daliPipeline_h pipeline, daliPipelineOutputs_h *out);

/** Pops the pipeline outputs from the pipeline's output queue.
 *
 * The outputs are ready for use on the provided stream.
 * When no longer used, the outputs must be freed by destroying the daliPipelineOutput object.
 *
 * This function works only with DALI_EXEC_IS_DYNAMIC.
 *
 * @param pipeline [in]  The pipeline whose outputs are to be obtained
 * @param out      [out] A pointer to the output handle. The handle is NULL if the function
 *                       reports an error.
 *
 * @return This function will report errors that occurred asynchronously when preparing the
 *         relevant data batch. If an error is reported, the output handle is NULL.
 */
DALI_API daliResult_t daliPipelinePopOutputsAsync(
  daliPipeline_h pipeline,
  daliPipelineOutputs_h *out,
  cudaStream_t stream);

/** Releases the pipeline outputs.
 *
 * @param pipeline [in]  The pipeline outputs which are being released.
 *
 * This function destroys the daliPipelineOutputObject. The availability of the outputs differs
 * between different executors.
 * If DALI_EXEC_IS_DYNAMIC is used, the outputs may be used until their handles are destroyed.
 * Otherwise, the outputs must not be used after this call has been made.
 *
 * @warning When NOT using DALI_EXEC_IS_DYNAMIC, the maximum number of live daliPipelineOutputs_h
 *          obtained from a single pipeline must not exceed the prefetch_queue_depth. Running the
 *          pipeline again after the maximum number of live output sets is reached is an undefined
 *          behavior.
 */
DALI_API daliResult_t daliPipelineOutputsDestroy(daliPipelineOutputs_h out);

typedef struct _DALIOperatorTrace {
  const char *operator_name;
  const char *trace;
  const char *value;
} daliOperatorTrace_t;

/** Gets all operator "traces" that were set when producing this set of outputs.
 *
 * @param outputs         [in]  The outputs
 * @param out_traces      [out] A pointer to the location where the pointer to the beginning of an
 *                              array of operator traces is stored.
 * @param out_trace_count [out] A pointer that receives the number of traces.
 *
 * The output array is valid until the `outputs` handle is destroyed.
 */
DALI_API daliResult_t daliPipelineOutputsGetTraces(
  daliPipelineOutputs_h outputs,
  const daliOperatorTrace_t **out_traces,
  int *out_trace_count);

/** Gets a single operator "trace", identified by operator instance name and a trace name.
 *
 * @param outputs       [in]  The outputs
 * @param out_trace     [out] A pointer which receives a ppointer to the trace.
 * @param operator_name [in]  The name of the operator whose trace is being obtained.
 * @param trace_name    [in]  The name of the trace.
 *
 * @retval DALI_SUCCESS           On success
 * @retval DALI_ERROR_INVALID_KEY When there's no trace that matches the names
 */
DALI_API daliResult_t daliPipelineOutputsGetTrace(
  daliPipelineOutputs_h outputs,
  const char **out_trace,
  const char *operator_name,
  const char *trace_name);

/** Gets index-th output.
 *
 * The handle returned by this function must be released with a call to daliTensorListDecRef.
 *
 * Unless the pipeline uses DALI_EXEC_IS_DYNAMIC flag, the returned tensor list must not be used
 * after the `outputs` handle is destroyed.
 *
 * @param outputs [in]  The pipeline outputs object
 * @param out     [out] A pointer to a TensorList handle
 * @param index   [in]  The index of the output to get a handle to.
 */
DALI_API daliResult_t daliPipelineOutputsGet(
  daliPipelineOutputs_h outputs,
  daliTensorList_h *out,
  int index);

/****************************************************************************/
/*** Checkpointing **********************************************************/
/****************************************************************************/

typedef struct _DALICheckpoint *daliCheckpoint_h;

typedef struct _DALICheckpointExternalBuffer {
  const char *data;
  size_t      size;
} daliCheckpointExternalBuffer_t;

/** Contains extra state stored alongside the pipeline checkpoint */
typedef struct _DALICheckpointExternalData {
  daliCheckpointExternalBuffer_t pipeline_data;
  daliCheckpointExternalBuffer_t iterator_data;
} daliCheckpointExternalData_t;

/** Gets the latest checkpoint.
 *
 * Gets the current state of the pipeline. It can be used later to restore the pipeline
 * to a state it was in at the time the checkpoint was obtained.
 *
 * The function returns a checkpoint as a handle, which must be destroyed with a call to
 * daliCheckpointDestroy.
 *
 * @param pipeline        [in]  The pipeline
 * @param out_checkpoint  [out] A pointer to the location where the checkpoint handle is stored
 * @param checkpoint_ext  [in]  An optional pointer to a structure with additional checkpoint data.
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_INVALID_OPERATION   The pipeline wasn't built or checkpointing is disabled.
 */
DALI_API daliResult_t daliPipelineGetCheckpoint(
  daliPipeline_h pipeline,
  daliCheckpoint_h *out_checkpoint,
  const daliCheckpointExternalData_t *checkpoint_ext);

/** Gets the checkpoint data, serialized as a byte buffer.
 *
 * Gets the serialized checkpoint.
 * The result is cached and remain valid until the checkpoint object is destroyed.
 *
 * @param pipeline    [in]  The pipeline
 * @param checkpoint  [in]  The checkpoint
 * @param out_data    [out] A pointer to the buffer containing the serialized checkpoint
 *                          The returned pointer remains valid until the checkpoint is destroyed.
 * @param out_size    [out] A pointer to the location where the checkpoint length is stored
 */
DALI_API daliResult_t daliPipelineSerializeCheckpoint(
  daliPipeline_h pipeline,
  daliCheckpoint_h checkpoint,
  const char **out_data,
  size_t *out_size);


/** Restores the state of the pipeline based on the checkpoint.
 *
 * @param pipeline    The pipeline whose state to restore. The pipeline must be identical to the one
 *                    from which the checkpoint was obtained.
 * @param checkpoint  The checkpoint containing the pipeline state.
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_INVALID_OPERATION The pipeline does not match the one from the checkpoint.
 */
DALI_API daliResult_t daliPipelineRestoreCheckpoint(
  daliPipeline_h pipeline,
  daliCheckpoint_h checkpoint);

/** Reconstitutes a checkpoint object from a byte buffer.
 *
 * @param pipeline                    [in]  The pipeline whose checkpoint is being deserialized.
 * @param checkpoint                  [out] The checkpoint.
 * @param serialized_checkpoint       [in]  A pointer to the beginning of the buffer containing the
 *                                          serialized checkpoint data.
 * @param serialized_checkpoint_size  [out] The length, in bytes, of the buffer.
 */
DALI_API daliResult_t daliPipelineDeserializeCheckpoint(
  daliPipeline_h pipeline,
  daliCheckpoint_h  *out_checkpoint,
  const char *serialized_checkpoint,
  size_t serialized_checkpoint_size);

/** Gets the external data associated with a checkpoint
 *
 * @param checkpoint    [in]  The checkpoint
 * @param out_ext_data  [out] A pointer to the location where the return value is stored.
 */
DALI_API daliResult_t daliCheckpointGetExternalData(
  daliCheckpoint_h checkpoint,
  daliCheckpointExternalData_t *out_ext_data);

/** Destroys a checkpoint object */
DALI_API daliResult_t daliCheckpointDestroy(daliCheckpoint_h checkpoint);

/****************************************************************************/
/*** Tensor and TensorList API **********************************************/
/****************************************************************************/

typedef struct _DALITensorDesc {
  /** The number of dimensions of the tensor.
   *
   * 0 denotes a scalar value. Negative values are invalid.
   */
  int             ndim;

  /** The shape of the tensor.
   *
   * The shape starts with the "slowest" dimension - a row-major 640x480 interleaved RGB image
   * would have the shape [480, 640, 3].
   *
   * The shape can be NULL if ndim == 0
   */
  const int64_t  *shape;

  /** The type of the elements of the tensor */
  daliDataType_t  dtype;

  /** The layout string of the tensor.
   *
   * A layout string consists of exactly `ndim` single-character axis labels. The entries in layout
   * correspond to the dimension in the shape. A row-major interleaved image
   * would have a layout "HWC"
   */
  const char     *layout;

  /** A pointer to the first element in the tensor.
   *
   * The data pointer can be NULL if the total volume of the tensor is 0.
   * It must not be NULL if ndim == 0.
   */
  void           *data;
} daliTensorDesc_t;

/** The specification of the buffer storage location */
typedef struct _DALIBufferPlacement {
  /** The type of the storage device (CPU or GPU). */
  daliStorageDevice_t device_type;

  /** CUDA device ordinal, as returned by CUDA runtime API.
   *
   * The value of this field is meaningful only if `device_type` is GPU or `pinned` is `true`.
   *
   * WARNING: The device_id returned by NVML (and thus, nvidia-smi) may be different.
   */
  int                 device_id;

  /** Whether the CPU storage is "pinned" - e.g. allocated with cudaMallocHost */
  daliBool            pinned;
} daliBufferPlacement_t;

/****************************************************************************/
/*** TensorList *************************************************************/
/****************************************************************************/

/** Creates a TensorList on the specified device */
DALI_API daliResult_t daliTensorListCreate(
  daliTensorList_h *out,
  daliBufferPlacement_t placement);

/** Changes the size of the tensor, allocating more data if necessary.
 *
 * @param num_samples   the number of samples in the batch
 * @param ndim          the number of dimensions of a sample
 * @param shapes        the concatenated shapes of the samples;
 *                      must contain num_samples*ndim extents
 * @param dtype         the element type
 * @param layout        a layout string describing the order of axes in each sample (e.g. HWC),
 *                      if NULL, and the TensorList's number of dimensions is equal to `ndim`,
 *                      then the current layout is kept;
 *                      if `layout` is an empty string, the tensor list's layout is cleared *
 */
DALI_API daliResult_t daliTensorListResize(
  daliTensorList_h tensor_list,
  int num_samples,
  int ndim,
  const int64_t *shapes,
  daliDataType_t dtype,
  const char *layout);

/** Attaches an externally allocated buffer to a TensorList.
 *
 * Attaches an externally allocated buffer and a deleter to a TensorList.
 * The deleter is called when the TensorList object is destroyed.
 *
 * The shape and sample offsets are used only during this function call and may be safely
 * disposed of after the function returns.
 *
 * @param tensor_list     the TensorList to attach the data to
 * @param num_samples     the number of samples in the list
 * @param ndim            the number of dimensions in the sample
 * @param shapes          the concatenated shapes of the samples;
 *                        must contain num_samples*ndim extents
 * @param dtype           the element type
 * @param layout          a layout string describing the order of axes in each sample (e.g. HWC),
 *                        if NULL, and the TensorList's number of dimensions is equal to `ndim`,
 *                        then the current layout is kept;
 *                        if `layout` is an empty string, the tensor list's layout is cleared
 * @param data            the pointer to the data buffer
 * @param sample_offsets  optional; the offsets, in bytes, of the samples in the batch from the
 *                        base pointer `data`; if NULL, the samples are assumed to be densely
 *                        packed, with the 0-th sample starting at the address `data`.
 * @param deleter         an optional deleter called when the buffer reference count goes to zero
 */
DALI_API daliResult_t daliTensorListAttachBuffer(
  daliTensorList_h tensor_list,
  int num_samples,
  int ndim,
  const int64_t *shapes,
  daliDataType_t dtype,
  const char *layout,
  void *data,
  const ptrdiff_t *sample_offsets,
  daliDeleter_t deleter);

/** Attaches externally allocated tensors to a TensorList.
 *
 * Attaches externally allocated buffers to a TensorList.
 * If provided, the deleters are called on all buffers when the samples are destroyed.
 *
 * The sample descriptors are used only during this function call and may be safely disposed of
 * after the function returns.
 *
 * @param tensor_list     the TensorList to attach the data to
 * @param num_samples     the new number of samples in the batch
 * @param ndim            the number of dimensions in each sample;
 *                        if num_samples > 0, this value can be set to -1 and the number of
 *                        dimensions will be taken from samples[0].ndim
 * @param dtype           the type of the element of the tensor;
 *                        if dtype is DALI_NO_TYPE, then the type is taken from samples[0].dtype
 * @param layout          a layout string describing the order of axes in each sample (e.g. HWC),
 *                        if NULL, the layout is taken from samples[0].layout; if it's still NULL,
 *                        the current layout is kept, if possible;
 *                        if `layout` is an empty string, the tensor list's layout is cleared
 * @param samples         the descriptors of the tensors to be attached to the TensorList;
 *                        the `ndim` and `dtype` of the samples must match and they must match the
 *                        values of `ndim` and `dtype` parameters; the layout must be either NULL
 *                        or match the `layout` argument (if provided).
 * @param sample_deleters optional deleters, one for each sample
 *
 * NOTE: If the sample_deleters specify the same object multiple times, its destructor must
 *       internally use reference counting to avoid multiple deletion.
 */
DALI_API daliResult_t daliTensorListAttachSamples(
  daliTensorList_h tensor_list,
  int num_samples,
  int ndim,
  daliDataType_t dtype,
  const char *layout,
  const daliTensorDesc_t *samples,
  const daliDeleter_t *sample_deleters);


/** Returns the placement of the TensorLists's underlying buffer.
 *
 * @param tensor_list   [in]  the TensorList whose buffer placement is queried
 * @param out_placement [out] a pointer to the location where the return value is stored.
 */
DALI_API daliResult_t daliTensorListGetBufferPlacement(
  daliTensorList_h tensor_list,
  daliBufferPlacement_t *out_placement);

/** Associates a stream with the TensorList.
 *
 * @param stream      an optional CUDA stream handle; if the handle pointer is NULL,
 *                    host-synchronous behavior is prescribed.
 * @param synchronize if true, the new stream (or host, if NULL), will be synchronized with the
 *                    currently associated stream
 */
DALI_API daliResult_t daliTensorListSetStream(
  daliTensorList_h tensor_list,
  const cudaStream_t *stream,
  daliBool synchronize
);

/** Gets the stream associated with the TensorList.
 *
 * @retval DALI_SUCCESS if the stream handle was stored in *out_stream
 * @retval DALI_NO_DATA if the tensor list is not associated with any stream
 *         error code otherwise
 */
DALI_API daliResult_t daliTensorListGetStream(
  daliTensorList_h tensor_list,
  cudaStream_t *out_stream
);

/** Gets the readiness event associated with the TensorList.
 *
 * @param tensor_list [in]  the tensor list whose ready event is to be obtained
 * @param out_event   [out] the pointer to the return value
 *
 * @retval DALI_SUCCESS if the ready event handle was stored in *out_event
 * @retval DALI_NO_DATA if the tensor list is not associated with a readiness event
 *         error code otherwise
 */
DALI_API daliResult_t daliTensorListGetReadyEvent(
  daliTensorList_h tensor_list,
  cudaEvent_t *out_event);

/** Gets the readiness event associated with the TensorList or creates a new one.
 *
 * @param tensor_list   [in]  the tensor list to associate an event with
 * @param out_event     [out] optional, the event handle
 *
 * The function ensures that a readiness event is associated with the tensor list.
 * It can also get the event handle, if the output parameter pointer is not NULL.
 * The function fails if the tensor list is not associated with a CUDA device.
 */
DALI_API daliResult_t daliTensorListGetOrCreateReadyEvent(
  daliTensorList_h tensor_list,
  cudaEvent_t *out_event);


/** Gets the shape of the tensor list
 *
 * @param tensor_list     [in]  the tensor list whose shape to obtain
 * @param out_num_samples [out] optional; the number of samples in the batch
 * @param out_ndim        [out] optional; the number of dimensions in a sample
 * @param out_shape       [out] optional; the pointer to the concatenated array of sample shapes;
 *                              contains (*out_num_samples) * (*out_ndim) elements
 *
 * The pointer returned in `out_shape` remains valid until the TensorList is destroyed or modified.
 * If the caller is not interested in some of the values, the pointers can be NULL.
 */
DALI_API daliResult_t daliTensorListGetShape(
  daliTensorList_h tensor_list,
  int *out_num_samples,
  int *out_ndim,
  const int64_t **out_shape);

/** Gets a layout string describing the samples in the TensorList.
 *
 * @param tensor_list [in]  the tensor list whose layout to obtain
 * @param out_layout  [out] a pointer to the place where a pointer to the layout string of
 *                          the samples in the tensor list is stored
 *
 * When present, the layout string consists of exactly `sample_ndim` single-character _axis labels_.
 * The layout does not contain the leading "sample" dimension (typically denoted as `N`),
 * for example, a batch of images would typically have a "HWC" layout.
 * The axis labels can be any character except the null character '\0'.
 * If there's no layout set, the returned pointer is NULL.
 *
 * The pointer remains valid until the tensor list is destroyed, cleared, resized or its layout
 * changed.
 */
DALI_API daliResult_t daliTensorListGetLayout(
  daliTensorList_h tensor_list,
  const char **out_layout);

/** Sets the layout of the samples in the TensorList.
 *
 * Sets the axis labels that describe the layout of the data in the TensorList. The layout must not
 * contain the leading sample dimension (typically `N`). For example, a batch of images would
 * typically have a layout "HWC".
 * If the layout string is NULL or empty, the layout is cleared; otherwise it must contain exactly
 * sample_ndim nonzero characters. The axis labels don't have to be unique.
 */
DALI_API daliResult_t daliTensorListSetLayout(
  daliTensorList_h tensor_list,
  const char *layout
);

/** Gets the "source info" metadata of a sample.
 *
 * Each sample can be associated with a "source info" string, which typically is the file name,
 * but can also contain an index in a container, key, etc.
 *
 * @param tensor_list     [in]  The tensor list
 * @param out_source_info [out] A pointer to the location where the pointer to the source_info
 *                              string is stored. On success, `*out_source_info` contains a pointer
 *                              to the beginning of a null-terminated string. If the sample doesn't
 *                              have associated source info, a NULL pointer is returned.
 * @param sample_idx      [in]  The index of a sample whose source info is queried.
 *
 * The return value is a string pointer. It is invalidated by destroying, clearing or resizing
 * the TensorList as well as by assigning a new source info.
 */
DALI_API daliResult_t daliTensorListGetSourceInfo(
  daliTensorList_h tensor_list,
  const char **out_source_info,
  int sample_idx);

/** Sets the "source info" metadata of a tensor in a list.
 *
 * A tensor can be associated with a "source info" string, which typically is the file name,
 * but can also contain an index in a container, key, etc.
 *
 * @param tensor_list [in]  The tensor list
 * @param sample_idx  [in]  The index of the sample, whose source info will is being set.
 * @param source_info [in]  A source info string (e.g. filename) to associate with the tensor.
 *                          Passing NULL is equivalent to passing an empty string.
 */
DALI_API daliResult_t daliTensorListSetSourceInfo(
  daliTensorList_h tensor_list,
  int sample_idx,
  const char *source_info);


/** Gets the tensor descriptor of the specified sample.
 *
 * @param tensor_list [in]  The tensor list
 * @param out_desc    [out] A pointer to a location where the descriptor is written.
 * @param sample_idx  [in]  The index of the sample, whose descriptor to get.
 *
 * The descriptor stored in `out_desc` contains pointers. These pointers are invalidated by
 * destroying, clearing or resizing the TensorList or re-attaching new data to it.
 */
DALI_API daliResult_t daliTensorListGetTensorDesc(
  daliTensorList_h tensor_list,
  daliTensorDesc_t *out_desc,
  int sample_idx);

/** Increments the reference count of the tensor list.
 *
 * @param tensor_list [in]  A handle to the tensor list.
 * @param new_count   [out] If not NULL, the incremented reference count is returned in *new_count.
 */
DALI_API daliResult_t daliTensorListIncRef(daliTensorList_h tensor_list, int *new_count);

/** Decrements the reference count of the tensor list.
 *
 * The handle is destroyed if the reference count reaches 0.
 * When the client code no longer needs the handle, it must call daliTensorDecRef.
 *
 *
 * @param tensor_list [in]  A handle to the tensor list.
 * @param new_count   [out] If not NULL, the incremented reference count is returned in *new_count.
 */
DALI_API daliResult_t daliTensorListDecRef(daliTensorList_h tensor_list, int *new_count);

/** Reads the current reference count of the tensor list.
 *
 * @param tensor_list [in]  A handle to the tensor list.
 * @param count       [out] The ouput parameter that receives the reference count.
 */
DALI_API daliResult_t daliTensorListRefCount(daliTensorList_h tensor_list, int *count);

/** Views a TensorList as a Tensor.
 *
 * Creates a new Tensor that points to the same data as the TensorList. The samples in the
 * TensorList must have a uniform shape and the data in the TensorList must be contiguous.
 *
 * The tensor holds a reference to the data in the TensorList - it is safe to destroy the
 * TensorList and continue using the resulting Tensor.
 *
 * @retval DALI_SUCCESS on success
 * @retval DALI_ERROR_INVALID_OPERATION if the data is not contiguous
 * @retval DALI_ERROR_INVALID_HANDLE    the tensor list handle is invalid
 * @return DALI_ERROR_INVALID_ARGUMENT  the tensor handle pointer is NULL
 * @return DALI_ERROR_OUT_OF_MEMORY
 */
DALI_API daliResult_t daliTensorListViewAsTensor(
  daliTensorList_h tensor_list,
  daliTensor_h *out_tensor);

/***************************************************************************/
/*** Tensor ****************************************************************/
/***************************************************************************/

/** Creates a Tensor on the specified device */
DALI_API daliResult_t daliTensorCreate(
  daliTensor_h *out,
  daliBufferPlacement_t placement);

/** Changes the size of the tensor, allocating more data if necessary.
 *
 * @param num_samples   the number of samples in the batch
 * @param ndim          the number of dimensions of a sample
 * @param shape         the shape of the tensor; can be NULL if ndim == 0
 * @param dtype         the element type
 * @param layout        a layout string describing the order of axes in the tensor (e.g. HWC),
 *                      if NULL, and the Tensor's number of dimensions is equal to `ndim`,
 *                      then the current layout is kept;
 *                      if `layout` is an empty string, the tensor's layout is cleared
 */
DALI_API daliResult_t daliTensorResize(
  daliTensor_h tensor,
  int ndim,
  const int64_t *shape,
  daliDataType_t dtype,
  const char *layout);

/** Attaches an externally allocated buffer to a Tensor.
 *
 * Attaches an externally allocated buffer and a deleter to a Tensor.
 * The deleter is called when the Tensor object is destroyed.
 *
 * The shape and layout are used only during this function call and may be safely
 * disposed of after the function returns.
 *
 * @param tensor          the Tensor to attach the data to
 * @param ndim            the number of dimensions in the sample
 * @param dtype           the element type
 * @param shape           the shape of the tensor; ndim extents; can be NULL if ndim == 0
 * @param layout          a layout string describing the order of axes in the tensor (e.g. HWC),
 *                        if NULL, and the Tensor's number of dimensions is equal to `ndim`,
 *                        then the current layout is kept;
 *                        if `layout` is an empty string, the tensor's layout is cleared
 * @param data            the pointer to the data buffer
 * @param deleter         the deleter to be called when the tensor is destroyed
 */
DALI_API daliResult_t daliTensorAttachBuffer(
  daliTensor_h tensor,
  int ndim,
  const int64_t *shape,
  daliDataType_t dtype,
  const char *layout,
  void *data,
  daliDeleter_t deleter);

/** Returns the placement of the Tensor's underlying buffer.
 *
 * @param tensor        [in]  the Tensor whose buffer placement is queried
 * @param out_placement [out] a pointer to the location where the return value is stored.
 */
DALI_API daliResult_t daliTensorGetBufferPlacement(
  daliTensor_h tensor,
  daliBufferPlacement_t *out_placement);

/** Associates a stream with the Tensor.
 *
 * @param stream      an optional CUDA stream handle; if the handle pointer is NULL,
 *                    host-synchronous behavior is prescribed.
 * @param synchronize if true, the new stream (or host, if NULL), will be synchronized with the
 *                    currently associated stream
 */
DALI_API daliResult_t daliTensorSetStream(
  daliTensor_h tensor,
  const cudaStream_t *stream,
  daliBool synchronize
);

/** Gets the stream associated with the Tensor.
 *
 * @retval DALI_SUCCESS if the stream handle was stored in *out_stream
 * @retval DALI_NO_DATA if the tensor is not associated with any stream
 *         error code otherwise
 */
DALI_API daliResult_t daliTensorGetStream(
  daliTensor_h tensor,
  cudaStream_t *out_stream
);

/** Gets the readiness event associated with the Tensor.
 *
 * @param tensor      [in]  the tensor list whose ready event is to be obtained
 * @param out_event   [out] the pointer to the return value
 *
 * @retval DALI_SUCCESS if the ready event handle was stored in *out_event
 * @retval DALI_NO_DATA if the tensor is not associated with a readiness event
 *         error code otherwise
 */
DALI_API daliResult_t daliTensorGetReadyEvent(
  daliTensor_h tensor,
  cudaEvent_t *out_event);

/** Gets the readiness event associated with the Tensor or creates a new one.
 *
 * @param tensor        [in]  the tensor to associate an event with
 * @param out_event     [out] optional, the event handle
 *
 * The function ensures that a readiness event is associated with the tensor.
 * It can also get the event handle, if the output parameter pointer is not NULL.
 * The function fails if the tensor is not associated with a CUDA device.
 */
DALI_API daliResult_t daliTensorGetOrCreateReadyEvent(
  daliTensor_h tensor,
  cudaEvent_t *out_event);


/** Gets the shape of the tensor
 *
 * @param tensor      [in]  the tensor whose shape to obtain
 * @param out_ndim    [out] optional; receives the number of dimensions
 * @param out_shape   [out] optional; receives the pointer to the shape (array of extents)
 *
 * The pointer returned in `out_shape` remains valid until the Tensor is destroyed or modified.
 * If the caller is not interested in some of the values, the pointers can be NULL.
 */
DALI_API daliResult_t daliTensorGetShape(
  daliTensor_h tensor,
  int *out_ndim,
  const int64_t **out_shape);

/** Gets a layout string describing the data in the Tensor.
 *
 * @param tensor      [in]  the tensor whose layout to obtain
 * @param out_layout  [out] a pointer to the place where a pointer to the layout string of
 *                          the samples in the tensor is stored
 *
 * When present, the layout string consists of exactly `ndim` single-character _axis labels_.
 * for example, an image would typically have a "HWC" layout.
 * The axis labels can be any character except the null character '\0'.
 * If there's no layout set, the returned pointer is NULL.
 *
 * The pointer remains valid until the tensor is destroyed, cleared, resized or its layout
 * changed.
 */
DALI_API daliResult_t daliTensorGetLayout(
  daliTensor_h tensor,
  const char **out_layout);

/** Sets the layout of the data in the Tensor.
 *
 * Sets the axis labels that describe the layout of the data in the Tensor.
 * If the layout string is NULL or empty, the layout is cleared; otherwise it must contain exactly
 * sample_ndim nonzero characters. The axis labels don't have to be unique.
 */
DALI_API daliResult_t daliTensorSetLayout(
  daliTensor_h tensor,
  const char *layout
);

/** Gets the "source info" metadata of a tensor.
 *
 * A tensor can be associated with a "source info" string, which typically is the file name,
 * but can also contain an index in a container, key, etc.
 *
 * @param tensor          [in]  The tensor
 * @param out_source_info [out] A pointer to the location where the pointer to the source_info
 *                              string is stored. On success, `*out_source_info` contains a pointer
 *                              to the beginning of a null-terminated string. If the sample doesn't
 *                              have associated source info, a NULL pointer is returned.
 *
 * The return value is a string pointer. It is invalidated by destroying, clearing or resizing
 * the Tensor as well as by assigning a new source info.
 */
DALI_API daliResult_t daliTensorGetSourceInfo(
  daliTensor_h tensor,
  const char **out_source_info);

/** Sets the "source info" metadata of a tensor.
 *
 * A tensor can be associated with a "source info" string, which typically is the file name,
 * but can also contain an index in a container, key, etc.
 *
 * @param tensor      [in]  The tensor
 * @param source_info [in]  A source info string (e.g. filename) to associate with the tensor.
 *                          Passing NULL is equivalent to passing an empty string.
 */
DALI_API daliResult_t daliTensorSetSourceInfo(
  daliTensor_h tensor,
  const char *source_info);

/** Gets the descriptor of the data in the tensor.
 *
 * @param tensor      [in]  The tensor
 * @param out_desc    [out] A pointer to a location where the descriptor is written.
 *
 * The descriptor stored in `out_desc` contains pointers. These pointers are invalidated by
 * destroying, clearing or resizing the Tensor or re-attaching new data to it.
 */
DALI_API daliResult_t daliTensorGetDesc(
  daliTensor_h tensor,
  daliTensorDesc_t *out_desc);

/** Increments the reference count of the tensor.
 *
 * @param tensor      [in]  A handle to the tensor.
 * @param new_count   [out] If not NULL, the incremented reference count is returned in *new_count.
 */
DALI_API daliResult_t daliTensorIncRef(daliTensor_h tensor, int *new_count);

/** Decrements the reference count of the tensor.
 *
 * The handle is destroyed if the reference count reaches 0.
 * When the client code no longer needs the handle, it must call daliTensorDecRef.
 *
 *
 * @param tensor      [in]  A handle to the tensor.
 * @param new_count   [out] If not NULL, the incremented reference count is returned in *new_count.
 */
DALI_API daliResult_t daliTensorDecRef(daliTensor_h tensor, int *new_count);

/** Reads the current reference count of the tensor.
 *
 * @param tensor      [in]  A handle to the tensor.
 * @param count       [out] The ouput parameter that receives the reference count.
 */
DALI_API daliResult_t daliTensorRefCount(daliTensor_h tensor, int *count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DALI_DALI_H_
