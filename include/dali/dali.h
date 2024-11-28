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

  /** A path to a file or other OS resource is invalid */
  DALI_ERROR_PATH_NOT_FOUND,
  /** An I/O operation failed */
  DALI_ERROR_IO_ERROR,

  /** A memory allocation failed */
  DALI_ERROR_OUT_OF_MEMORY = DALI_ERROR + 0x100,

  /** Internal error - logic error in DALI code */
  DALI_ERROR_INTERNAL = DALI_ERROR + 0x200,
  /** The library was not properly initialized */
  DALI_ERROR_NOT_INITIALIZED,
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
 * Succesful calls do not overwrite the value.
 */
DALI_API daliResult_t daliGetLastError();

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
DALI_API const char *daliGetErrorName(daliResult_t error);

/** Returns a human-readable description of a given error.
 *
 * The value is a pointer to a string literal. It's not invalidated other than by unloading DALI.
 */
DALI_API const char *daliGetErrorDescription(daliResult_t error);


/** Initializes DALI or increments initialization count. */
DALI_API daliResult_t daliInit();

/** Decrements initialization counts and shuts down the library when the count reaches 0.
 *
 * Calling this function is optional. DALI will be shut down automatically when the program exits.
 */
DALI_API daliResult_t daliShutdown();

DALI_API daliResult_t daliPreallocateDeviceMemory2(size_t bytes, int device_id);

/** Allocates `bytes` bytes of device memory on device `device_id`.
 *
 * The function works by allocating and immediately freeing the specified amount of device
 * memory. This will typically release the memory back to DALI's memory pool, speeding up
 * subsequent allocations.
 */
inline daliResult_t daliPreallocateDeviceMemory(size_t bytes, int device_id) {
  return daliPreallocateDeviceMemory2(bytes, device_id);
}

DALI_API daliResult_t daliPreallocatePinnedMemory2(size_t bytes);

/** Allocates `bytes` bytes of device-accessible host memory.
 *
 * The function works by allocating and immediately freeing the specified amount of pinned
 * memory. This will typically release the memory back to DALI's memory pool, speeding up
 * subsequent allocations.
 */
inline daliResult_t daliPreallocatePinnedMemory(size_t bytes) {
  return daliPreallocatePinnedMemory2(bytes);
}

DALI_API daliResult_t daliReleaseUnusedMemory2();

/** Releases unused memory from DALI memory pools to the operating system.
 *
 * NOTE: Some of the memory pool implementations allocate memory from the OS in large chunks.
 *       If the chunk is occupied by even a tiny allocation, it will not be freed by this function.
 */
inline daliResult_t daliReleaseUnusedMemory() {
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
} daliExecType_t;

typedef struct _DALIVersion {
  int16_t major, minor;
  int32_t patch;
} daliVersion_t;


/** DALI Pipeline construction parameters */
typedef struct _DALIPipelineParams {
  /** The version of this structure */
  daliVersion_t version;

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
DALI_API daliResult_t daliPipelineCreate(
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

/** Executes the pipeline several times, to fill the buffer queues.
 *
 * NOTE: ExternalSource operators will need to be fet an appropriate number of times before this
 *       function can succeeed. Please check the required feed count by calling
 *       `daliPipelineFeedCount`.
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_OUT_OF_RANGE        if `input_name` is not a valid name of an input of the
 *                                        pipeline
 */
DALI_API daliResult_t daliPipelinePrefetch(daliPipeline_h pipeline);

/** Gets the required feed count for the specified input of the pipeline.
 *
 * @param pipeline        [in]  The pipeline
 * @param out_feed_count  [out] The number of batches to feed into the specified input before
 *                              `daliPipelinePrefetch` can be called.
 * @param input_name      [in]  The name of the input.
 *
 * @retval DALI_SUCCESS
 * @retval DALI_ERROR_OUT_OF_RANGE        if `input_name` is not a valid name of an input of the
 *                                        pipeline
 */
DALI_API daliResult_t daliPipelineGetFeedCount(
  daliPipeline_h pipeline,
  int *out_feed_count,
  const char *input_name);

typedef enum _DALIFeedInputFlags {
  /** Do not make a copy of the input, use it directly instead.
   *
   * When daliTensorList_h is passed to daliFeedInput, a reference count is incremented
   */
  DALI_FEED_INPUT_NO_COPY = 1,
} daliFeedInputFlags_t;

/** Feeds the input `input_name` with data from `input_data`.
 *
 * @param pipeline      the pipeline
 * @param input_name    the name of the input
 * @param input_data    the tensor list containing the data
 * @param data_id       an identifier of this data batch
 * @param options
 */
DALI_API daliResult_t daliPipelineFeedInput(
  daliPipeline_h pipeline,
  const char *input_name,
  daliTensorList_h input_data,
  const char *data_id,
  daliFeedInputFlags_t options,
  const cudaStream_t *stream);

/** Gets the number of pipeline outputs */
DALI_API daliResult_t daliPipelineGetOutputCount(daliPipeline_h pipeline, int *out_count);

/** Gets the number of pipeline outputs */
DALI_API daliResult_t daliPipelineGetOutputDesc(
  daliPipeline_h pipeline,
  daliPipelineOutputDesc_t *out_desc,
  int index);


/** Pops the pipeline outputs from the pipeline's output queue.
 *
 * The outputs are ready for use on any stream.
 * When no longer used, the outputs should be freed by destroying the daliPipelineOutput object.
 *
 * @return This function will report errors that occurred asynchronously when preparing the
 *         relevant data batch.
 *
 */
DALI_API daliResult_t daliPipelinePopOutputs(daliPipeline_h pipeline, daliPipelineOutputs_h *out);

/** Pops the pipeline outputs from the pipeline's output queue.
 *
 * The outputs are ready for use on the provided stream.
 * When no longer used, the outputs should be freed by destroying the daliPipelineOutput object.
 *
 * This function works only with DALI_EXEC_IS_DYNAMIC.
 *
 * @return This function will report errors that occurred asynchronously when preparing the
 *         relevant data batch.
 */
DALI_API daliResult_t daliPipelinePopOutputsAsync(
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
DALI_API daliResult_t daliPipelineOutputsDestroy(daliPipelineOutputs_h out);

/** Gets index-th output.
 *
 * The handle returned by this function must be released with a call to daliTensorListDecRef
 */
DALI_API daliResult_t daliPipelineOutputsGet(
  daliPipelineOutputs_h outputs,
  daliTensorList_h *out,
  int index);

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
  int64_t        *shape;

  /** The type of the elements of the tensor */
  daliDataType_t  dtype;

  /** The layout string of the tensor.
   *
   * A layout string consists of exactly `ndim` single-character axis labels. The entries in layout
   * correspond to the dimension in the shape. A row-major interleaved image
   * would have a layout "HWC"
   */
  char           *layout;

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
   * WARNING: The device_id returned by NVML (and thus, nvidia-smi) may be different.
   */
  int                 device_id;

  /** Whether the CPU storage is "pinned" - e.g. allocated with cudaMallocHost */
  daliBool            pinned;
} daliBufferPlacement_t;

/** Creates a TensorList on the specified device */
DALI_API daliResult_t daliTensorListCreate(
  daliTensorList_h *out,
  daliBufferPlacement_t placement);

/** Changes the size of the tensor, allocating more data if necessary.
 *
 * @param num_samples   the number of samples in the batch
 * @param ndim          the number of dimensions of a sample
 * @param dtype         the element type
 * @param shapes        the concatenated shapes of the samples;
 *                      must contain num_samples*ndim extents
 */
DALI_API daliResult_t daliTensorListResize(
  daliTensorList_h tensor_list,
  int num_samples,
  int ndim,
  daliDataType_t dtype,
  const int64_t *shapes);

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
 * @param dtype           the element type
 * @param shapes          the concatenated shapes of the samples;
 *                        must contain num_samples*ndim extents
 * @param data            the pointer to the data buffer
 * @param sample_offsets  optional; the offsets, in bytes, of the samples in the batch from the
 *                        base pointer `data`; if NULL, the samples are assumed to be densely
 *                        packed, with the 0-th sample starting at the address `data`.
 */
DALI_API daliResult_t daliTensorListAttachBuffer(
  daliTensorList_h tensor_list,
  int num_samples,
  int ndim,
  daliDataType_t dtype,
  const int64_t *shapes,
  void *data,
  const ptrdiff_t *sample_offsets,
  daliDeleter_t deleter);

/** Attaches externally allocated tensors to a TensorList.
 *
 * Attaches externally allocated buffers to a TensorList.
 * If provided, the deleters are called on all buffers when the samples are destroyed.
 *
 * The shape and sample offsets are used only during this function call and may be safely
 * disposed of after the function returns.
 *
 * @param tensor_list     the TensorList to attach the data to
 * @param num_samples     the new number of samples in the batch
 * @param ndim            the number of dimensions in each sample;
 *                        if num_samples > 0, this value can be set to -1 and the number of
 *                        dimensions will be taken from samples[0].ndim
 * @param dtype           the type of the element of the tensor;
 *                        if dtype is DALI_NO_TYPE, then the type is taken from samples[0].dtype;
 *                        if set, the dtype in the samples can be left at -1
 * @param samples         the descriptors of the tensors to be attached to the TensorList;
 *                        the `ndim` and `dtype` of the samples must match and they must match the
 *                        values of `ndim` and `dtype` parameters.
 * @param sample_deleters the deleters, one for each sample (or NULL)
 *
 * NOTE: If the sample_deleters specify the same object multiple times, its destructor must
 *       internally use reference counting to avoid multiple deletion.
 */
DALI_API daliResult_t daliTensorListAttachSamples(
  daliTensorList_h tensor_list,
  int num_samples,
  int ndim,
  daliDataType_t dtype,
  const daliTensorDesc_t *samples,
  const daliDeleter_t *sample_deleters);


/** Associates a stream with the TensorList.
 *
 * @param tensor_list   [in]  the TensorList whose buffer placement is queried
 * @param out_placement [out] a pointer to a place where the return value is stored.
 */
DALI_API daliResult_t daliTensorListGetBufferPlacement(
  daliTensorList_h tensor_list,
  daliBufferPlacement_t *out_placement);

/** Associates a stream with the TensorList.
 *
 * @param stream      an optional CUDA stream handle; if the handle poitner is NULL,
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
 * @param tensor_list [in]  the tenosr list whose ready event is to be obtained
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
 * @param tensor_list   [in]  the tensor list to associate an even twith
 * @param out_event     [out] optional, the event handle
 *
 * The function ensures that a readiness event is associated with the tensor list.
 * It can also get the event handle, if the output parameter pointer is not NULL.
 */
DALI_API daliResult_t daliTensorListGetOrCreateReadyEvent(
  daliTensorList_h tensor_list,
  cudaEvent_t *out_event);


/** Gets the shape of the tensor list
 *
 * @param tensor_list     [in]  the tensor list whose shape obtain
 * @param out_num_samples [out] optional; the number of samples in the batch
 * @param out_ndim        [out] optional; the number of dimensions in a sample
 * @param out_shape       [out] optional; the pointer to the concatenated array of sample shapes;
 *                              contains (*out_num_samples) * (*out_ndim) elements
 *
 * @retval DALI_SUCCESS T
 *
 * The pointer returned in `out_shape` remains valid until the TensorList is destroyed or modified.
 * If the caller is not intersted in some of the values, the pointers can be NULL.
 */
DALI_API daliResult_t daliTensorListGetShape(
  daliTensorList_h tensor_list,
  int *out_num_samples,
  int *out_ndim,
  const int64_t **out_shape);

/** Gets a layout string describing the samples in the TensorList.
 *
 * When present, the layout string consists of exactly `sample_ndim` single-character _axis labels_.
 * The layout does not contain the leading "sample" dimension (typically denoted as `N`),
 * for example, a batch of images would typically have a "HWC" layout.
 * The axis labels can be any character except the '\0'.
 * If there's no layout set, the returned pointer is NULL and the function returns DALI_NO_DATA
 * status.
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

/** Gets the "source info" metadata of a sample */
DALI_API daliResult_t daliTensorListGetSourceInfo(
  daliTensorList_h tensor_list,
  const char **out_source_info,
  int sample_idx);

/** Gets the tensor descriptor of the specified sample.
 *
 * @param tensor_list [in] The tensor list
 * @param out_desc    [out] A poitner to a decriptor filled by this funciton.
 * @param sample_idx  [in] The index of the sample, whose descriptor to get.
 *
 * The descriptor stored in `out_desc` contains pointers. These pointers are invalidated by
 * clearing or resizing the TensorList or re-attaching new data to it.
 */
DALI_API daliResult_t daliTensorListGetTensor(
  daliTensorList_h tensor_list,
  daliTensorDesc_t *out_desc,
  int sample_idx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DALI_DALI_H_