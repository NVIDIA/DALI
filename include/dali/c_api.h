// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_C_API_H_
#define DALI_C_API_H_

#include <cuda_runtime_api.h>
#include <inttypes.h>
#include "dali/core/api_helper.h"

// Trick to bypass gcc4.9 old ABI name mangling used by TF
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Handle for DALI C-like API.
 *
 * @note Beware, the C API is just C-like API for handling some mangling issues and
 * it can throw exceptions.
 */
typedef struct {
  void *pipe;
  void *ws;
  void *batch_sizes_map;     /// @see batch_size_map_t
  cudaStream_t copy_stream;  /// Stream to perform copy operations on
} daliPipelineHandle;

typedef enum {
  CPU = 0,
  GPU = 1
} device_type_t;

typedef enum {
  DALI_BACKEND_CPU = 0,
  DALI_BACKEND_GPU = 1,
  DALI_BACKEND_MIXED = 2
} dali_backend_t;

typedef enum {
  DALI_NO_TYPE  = -1,
  DALI_UINT8    =  0,
  DALI_UINT16   =  1,
  DALI_UINT32   =  2,
  DALI_UINT64   =  3,
  DALI_INT8     =  4,
  DALI_INT16    =  5,
  DALI_INT32    =  6,
  DALI_INT64    =  7,
  DALI_FLOAT16  =  8,
  DALI_FLOAT    =  9,
  DALI_FLOAT64  =  10,
  DALI_BOOL     =  11,
} dali_data_type_t;


/*
 * Need to keep that in sync with ReaderMeta from operator.h
 */
typedef struct {
  int64_t epoch_size;          // raw epoch size
  int64_t epoch_size_padded;   // epoch size with the padding at the end
  int number_of_shards;        // number of shards
  int shard_id;                // shard id of given reader
  int pad_last_batch;          // if given reader should pad last batch
  int stick_to_shard;          // if given reader should stick to its shard
} daliReaderMetadata;


/*
 * Need to keep that in sync with ExecutorMeta from executor.h
 */
typedef struct {
  char *operator_name;         // operator name, user need to free the memory
  size_t out_num;              // number of the operator outputs
  size_t *real_size;           // real size of the operator output, user need to free the memory
  size_t *max_real_size;       // the biggest size of the tensor in the batch
  size_t *reserved;            // reserved size of the operator output, user need to free the memory
  size_t *max_reserved;        // the biggest reserved memory size for the tensor in the batch
} daliExecutorMetadata;

/**
 * @brief DALI initialization
 *
 * Call this function to initialize DALI backend. It shall be called once per process.
 * Along with this, you'll need to call @see daliInitOperatorsLib() function from
 * `operators.h` file, to initialize whole DALI.
 * In the unlikely event you'd like to use only Pipeline and Executor (no Operators),
 * you may pass on calling @see daliInitOperatorsLib()
 */
DLL_PUBLIC void daliInitialize();

/// @{
/**
 * @brief Create DALI pipeline. Setting max_batch_size,
 * num_threads or device_id here overrides
 * values stored in the serialized pipeline.
 * When separated_execution is equal to 0, prefetch_queue_depth is considered,
 * gpu_prefetch_queue_depth and cpu_prefetch_queue_depth are ignored.
 * When separated_execution is not equal to 0, cpu_prefetch_queue_depth and
 * gpu_prefetch_queue_depth are considered and prefetch_queue_depth is ignored.
 */
DLL_PUBLIC void daliCreatePipeline(daliPipelineHandle *pipe_handle, const char *serialized_pipeline,
                                   int length, int max_batch_size, int num_threads, int device_id,
                                   int separated_execution, int prefetch_queue_depth,
                                   int cpu_prefetch_queue_depth, int gpu_prefetch_queue_depth,
                                   int enable_memory_stats);

/**
 * Convenient overload. Use it, if the Pipeline should inherit its parameters
 * from serialized pipeline.
 */
DLL_PUBLIC void daliDeserializeDefault(daliPipelineHandle *pipe_handle,
                                       const char *serialized_pipeline,
                                       int length);
/// }@
/// @{

enum {
  DALI_ext_default = 0,
  /**
   * If memory transfer should be synchronous - applies to GPU memory
   */
  DALI_ext_force_sync = (1 << 0),

  /**
   * If provided CPU memory is page-locked
   */
  DALI_ext_pinned = (1 << 1),

  /**
   * If provided, a CUDA copy kernel will be used to feed external source instead of cudaMemcpyAsync
   * Only relevant when the input is either pinned host memory or device memory
   */
  DALI_use_copy_kernel = (1 << 2),

  /**
   * Override the `no_copy` specified for given External Source and force the data to be copied.
   */
  DALI_ext_force_copy = (1 << 3),

  /**
   * Override the `no_copy` specified for given External Source and pass the data directly to the
   * Pipeline.
   */
  DALI_ext_force_no_copy = (1 << 4),
};

/**
 * @brief Set the batch size for the upcoming call to `daliSetExternalInput*(...)`
 *
 * @param pipe_handle Pointer to pipeline handle
 * @param name Pointer to a null-terminated byte string with the name of the External Source
 *             to be fed
 * @param batch_size Batch size of the data
 */
DLL_PUBLIC void daliSetExternalInputBatchSize(daliPipelineHandle *pipe_handle, const char *name,
                                              int batch_size);

/**
 * @brief Feed the data to ExternalSource as contiguous memory.
 *
 * When calling this function, you need to provide a CUDA stream, which will be used when
 * copying data onto GPU. This function is asynchronous, so it's your responsibility to
 * synchronize on a provided CUDA stream.
 *
 * Keep in mind, that for the special case, where the data exists on the CPU and the
 * ExternalSource's Backend in also a CPU, stream is not needed - feel free to pass
 * the default stream.
 *
 * A convenience, synchronous, overload function is provided,
 * which handles the stream synchronization.
 *
 * If `daliSetExternalInputBatchSize` has been called prior to this function, given batch size
 * is assumed. Otherwise, the function will default to max batch size.
 * @see daliSetExternalInputBatchSize
 * @see daliCreatePipeline
 *
 * @param pipe_handle Pointer to pipeline handle
 * @param name Pointer to a null-terminated byte string with the name of the External Source
 *             to be fed
 * @param device Device of the supplied memory.
 * @param data_ptr Pointer to contiguous buffer containing all samples
 * @param data_type Type of the provided data
 * @param shapes Pointer to an array containing shapes of all samples concatenated one after
 *              another. Should contain batch_size * sample_dim elements.
 * @param sample_dim The dimensionality of a single sample.
 * @param layout_str Optional layout provided as a pointer to null-terminated byte string.
 *                   Can be set to NULL.
 * @param stream CUDA stream to use when copying the data onto GPU. Remember to synchronize on the
 *               provided stream.
 * @param flags Extra flags, check DALI_ext_* and DALI_use_copy_kernel flags
 */
DLL_PUBLIC void
daliSetExternalInputAsync(daliPipelineHandle *pipe_handle, const char *name,
                          device_type_t device, const void *data_ptr,
                          dali_data_type_t data_type, const int64_t *shapes,
                          int sample_dim, const char *layout_str,
                          cudaStream_t stream, unsigned int flags);

DLL_PUBLIC void
daliSetExternalInput(daliPipelineHandle *pipe_handle, const char *name,
                     device_type_t device, const void *data_ptr,
                     dali_data_type_t data_type, const int64_t *shapes,
                     int sample_dim, const char *layout_str, unsigned int flags);
///@}
///@{

/**
 * @brief Feed the data to ExternalSource as a set of separate buffers.
 *
 * When calling this function, you need to provide a CUDA stream, which will be used when
 * copying data onto GPU. This function is asynchronous, so it's your responsibility to
 * synchronize on a provided CUDA stream.
 *
 * Keep in mind, that for the special case, where the data exists on the CPU and the
 * ExternalSource's Backend in also a CPU, stream is not needed - feel free to pass
 * the default stream.
 *
 * A convenience, synchronous, overload function is provided,
 * which handles the stream synchronization.
 *
 * If `daliSetExternalInputBatchSize` has been called prior to this function, given batch size
 * is assumed. Otherwise, the function will default to max batch size.
 * @see daliSetExternalInputBatchSize
 * @see daliCreatePipeline
 *
 * @param pipe_handle Pointer to pipeline handle
 * @param name Pointer to a null-terminated byte string with the name of the External Source
 *             to be fed
 * @param device Device of the supplied memory.
 * @param data_ptr Pointer to an array containing batch_size pointers to separate Tensors.
 * @param data_type Type of the provided data
 * @param shapes Pointer to an array containing shapes of all samples concatenated one after
 *              another. Should contain batch_size * sample_dim elements.
 * @param sample_dim The dimensionality of a single sample.
 * @param layout_str Optional layout provided as a pointer to null-terminated byte string.
 *                   Can be set to NULL.
 * @param stream CUDA stream to use when copying the data onto GPU. Remember to synchronize on the
 *               provided stream.
 * @param flags Extra flags, check DALI_ext_force_sync, DALI_ext_pinned, DALI_use_copy_kernel
 */
DLL_PUBLIC void
daliSetExternalInputTensorsAsync(daliPipelineHandle *pipe_handle, const char *name,
                                 device_type_t device, const void *const *data_ptr,
                                 dali_data_type_t data_type, const int64_t *shapes,
                                 int64_t sample_dim, const char *layout_str,
                                 cudaStream_t stream, unsigned int flags);

DLL_PUBLIC void
daliSetExternalInputTensors(daliPipelineHandle *pipe_handle, const char *name,
                            device_type_t device, const void *const *data_ptr,
                            dali_data_type_t data_type, const int64_t *shapes,
                            int64_t sample_dim, const char *layout_str, unsigned int flags);
///@}

/**
 * @brief Start the execution of the pipeline.
 */
DLL_PUBLIC void daliRun(daliPipelineHandle *pipe_handle);

/**
 * @brief Schedule first runs to fill buffers for Executor with UniformQueue policy.
 */
DLL_PUBLIC void daliPrefetchUniform(daliPipelineHandle *pipe_handle, int queue_depth);

/**
 * @brief Schedule first runs to fill buffers for Executor with SeparateQueue policy.
 */
DLL_PUBLIC void daliPrefetchSeparate(daliPipelineHandle *pipe_handle,
                                     int cpu_queue_depth, int gpu_queue_depth);

/**
 * @brief Wait until the output of the pipeline is ready.
 * Releases previously returned buffers.
 */
DLL_PUBLIC void daliOutput(daliPipelineHandle *pipe_handle);

/**
 * @brief Wait until the output of the pipeline is ready.
 * Doesn't release previously returned buffers.
 */
DLL_PUBLIC void daliShareOutput(daliPipelineHandle *pipe_handle);

/**
 * @brief Releases buffer returned by last daliOutput call.
 */
DLL_PUBLIC void daliOutputRelease(daliPipelineHandle *pipe_handle);

/**
 * @brief Returns 1 if the the output batch stored at position `n` in the pipeline can
 * be represented as dense, uniform tensor. Otherwise 0.
 *
 * This function may only be called after
 * calling Output function.
 */
DLL_PUBLIC int64_t daliOutputHasUniformShape(daliPipelineHandle *pipe_handle, int i);

/**
 * @brief Return the shape of the output tensor stored at position `n` in the pipeline.
 * Valid only if daliOutputHasUniformShape() returns 1.
 *
 * This function may only be called after
 * calling Output function.
 * @remarks Caller is responsible to 'free' the memory returned
 */
DLL_PUBLIC int64_t *daliShapeAt(daliPipelineHandle *pipe_handle, int n);

/**
 * @brief Return the type of the output tensor
 * stored at position `n` in the pipeline.
 * This function may only be called after
 * calling Output function.
 */
DLL_PUBLIC dali_data_type_t daliTypeAt(daliPipelineHandle *pipe_handle, int n);

/**
 * @brief Return the shape of the 'k' output tensor from tensor list
 * stored at position `n` in the pipeline.
 * This function may only be called after
 * calling Output function.
 * @remarks Caller is responsible to 'free' the memory returned
 */
DLL_PUBLIC int64_t *daliShapeAtSample(daliPipelineHandle *pipe_handle, int n, int k);

/**
 * @brief Return the number of tensors in the tensor list
 * stored at position `n` in the pipeline.
 */
DLL_PUBLIC size_t daliNumTensors(daliPipelineHandle *pipe_handle, int n);

/**
 * @brief Return the number of all elements in the tensor list
 * stored at position `n` in the pipeline.
 */
DLL_PUBLIC size_t daliNumElements(daliPipelineHandle *pipe_handle, int n);

/**
 * @brief Return the size of the tensor list
 * stored at position `n` in the pipeline.
 */
DLL_PUBLIC size_t daliTensorSize(daliPipelineHandle *pipe_handle, int n);

/**
 * @brief Return maximum number of dimensions from all tensors
 * from the tensor list stored at position `n` in the pipeline.
 */
DLL_PUBLIC size_t daliMaxDimTensors(daliPipelineHandle *pipe_handle, int n);

/**
 * @brief Returns number of DALI pipeline outputs
 */
DLL_PUBLIC unsigned daliGetNumOutput(daliPipelineHandle *pipe_handle);

/**
 * @brief Returns a string indicating name of the output given by id.
 * @remark The returned pointer is invalidated after calling `daliDeletePipeline(pipe_handle)`.
 */
DLL_PUBLIC const char *daliGetOutputName(daliPipelineHandle *pipe_handle, int id);

/**
 * @brief Returns device_type_t indicating device backing pipeline output given by id
 */
DLL_PUBLIC device_type_t daliGetOutputDevice(daliPipelineHandle *pipe_handle, int id);

/**
 * @brief Copy the output batch stored at position `output_idx` in the pipeline.
 * @remarks If the pipeline output is TensorList then it needs to be dense
 * @param pipe_handle Pointer to pipeline handle
 * @param dst Pointer to the destination buffer where the data will be copied
 * @param output_idx index of the pipeline output
 * @param dst_type Device type associated with the destination buffer (0 - CPU, 1 - GPU)
 * @param stream CUDA stream to use when copying the data to/from the GPU.
 * @param flags Extra flags, check DALI_ext_force_sync, DALI_use_copy_kernel
 */

DLL_PUBLIC void
daliOutputCopy(daliPipelineHandle *pipe_handle, void *dst, int output_idx, device_type_t dst_type,
               cudaStream_t stream, unsigned int flags);

/**
 * @brief Copy the samples in output stored at position `output_idx` in the pipeline
 *        to scattered memory locations.
 * @param pipe_handle Pointer to pipeline handle
 * @param dsts Pointers to the destination buffers where each sample will be copied.
 *        A nullptr dst pointer for a sample will discard that sample.
 * @param output_idx index of the pipeline output
 * @param dst_type Device type associated with the destination buffer (0 - CPU, 1 - GPU)
 * @param stream CUDA stream to use when copying the data to/from the GPU.
 * @param flags Extra flags, check DALI_ext_force_sync, DALI_use_copy_kernel
 */
DLL_PUBLIC void daliOutputCopySamples(daliPipelineHandle *pipe_handle, void **dsts, int output_idx,
                                      device_type_t dst_type, cudaStream_t stream,
                                      unsigned int flags);

/**
 * @brief DEPRECATED API: use daliOutputCopy instead
 */
DLL_PUBLIC void
daliCopyTensorNTo(daliPipelineHandle *pipe_handle, void *dst, int n, device_type_t dst_type,
                  cudaStream_t stream, int non_blocking);

/**
 * @brief DEPRECATED API: use daliOutputCopy instead
 */
DLL_PUBLIC void
daliCopyTensorListNTo(daliPipelineHandle *pipe_handle, void *dst, int output_id,
                      device_type_t dst_type, cudaStream_t stream, int non_blocking);

/**
 * @brief Delete the pipeline object.
 */
DLL_PUBLIC void daliDeletePipeline(daliPipelineHandle *pipe_handle);

/**
 * @brief Load plugin library
 */
DLL_PUBLIC void daliLoadLibrary(const char *lib_path);

/**
 * @brief Returns the named reader metadata
 *  @param reader_name Name of the reader to query
 *  @param meta Pointer to metadata to be filled by the function
 */
DLL_PUBLIC void daliGetReaderMetadata(daliPipelineHandle* pipe_handle, const char *reader_name,
                                      daliReaderMetadata* meta);

/**
 * @brief Returns the backend of the operator with a given \p operator_name
 * @param operator_name Name of the operator to query
 */
DLL_PUBLIC dali_backend_t daliGetOperatorBackend(daliPipelineHandle* pipe_handle,
                                                 const char *operator_name);

/**
 * @brief Obtains the executor statistics
 *  @param operator_meta Pointer to the memory allocated by the function with operator_meta_num
 *                       number of metadata entries. To free returned metadata use
 *                       `daliFreeExecutorMetadata` function
 *  @param operator_meta_num Pointer to the variable which will tell how many meta entries
 *                           (operators) have been files
 */
DLL_PUBLIC void daliGetExecutorMetadata(daliPipelineHandle* pipe_handle,
                                        daliExecutorMetadata **operator_meta,
                                        size_t *operator_meta_num);

/**
 * @brief Frees executor metadata obtained from daliGetExecutorMetadata
 *  @param operator_meta Pointer to the memory with metadata allocated by the
 *                       `daliGetExecutorMetadata`
 *  @param operator_meta_num Number of metadata entries provided by `daliGetExecutorMetadata`
 */
DLL_PUBLIC void daliFreeExecutorMetadata(daliExecutorMetadata *operator_meta,
                                         size_t operator_meta_num);

#ifdef __cplusplus
}
#endif

#endif  // DALI_C_API_H_
