// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
  cudaStream_t copy_stream;  /// Stream to perform copy operations on
} daliPipelineHandle;

typedef enum {
  CPU = 0,
  GPU = 1
} device_type_t;

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
 * @brief Create DALI pipeline. Setting batch_size,
 * num_threads or device_id here overrides
 * values stored in the serialized pipeline.
 * When separated_execution is equal to 0, prefetch_queue_depth is considered,
 * gpu_prefetch_queue_depth and cpu_prefetch_queue_depth are ignored.
 * When separated_execution is not equal to 0, cpu_prefetch_queue_depth and
 * gpu_prefetch_queue_depth are considered and prefetch_queue_depth is ignored.
 */
DLL_PUBLIC void daliCreatePipeline(daliPipelineHandle *pipe_handle,
                                   const char *serialized_pipeline,
                                   int length,
                                   int batch_size,
                                   int num_threads,
                                   int device_id,
                                   int separated_execution,
                                   int prefetch_queue_depth,
                                   int cpu_prefetch_queue_depth,
                                   int gpu_prefetch_queue_depth,
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
 * @param sync Whether to block until the provided data is copied to the internal DALI buffer
 * @param is_pinned Wheather the provided memory is page-locked (pinned)
 */
DLL_PUBLIC void
daliSetExternalInputAsync(daliPipelineHandle *pipe_handle, const char *name,
                          device_type_t device, const void *data_ptr,
                          dali_data_type_t data_type, const int64_t *shapes,
                          int sample_dim, const char *layout_str,
                          cudaStream_t stream, int sync, int is_pinned);

DLL_PUBLIC void
daliSetExternalInput(daliPipelineHandle *pipe_handle, const char *name,
                     device_type_t device, const void *data_ptr,
                     dali_data_type_t data_type, const int64_t *shapes,
                     int sample_dim, const char *layout_str, int is_pinned);
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
 * @param sync Whether to block until the provided data is copied to the internal DALI buffer
 * @param is_pinned Wheather the provided memory is page-locked (pinned)
 */
DLL_PUBLIC void
daliSetExternalInputTensorsAsync(daliPipelineHandle *pipe_handle, const char *name,
                                 device_type_t device, const void *const *data_ptr,
                                 dali_data_type_t data_type, const int64_t *shapes,
                                 int64_t sample_dim, const char *layout_str,
                                 cudaStream_t stream, int sync, int is_pinned);

DLL_PUBLIC void
daliSetExternalInputTensors(daliPipelineHandle *pipe_handle, const char *name,
                            device_type_t device, const void *const *data_ptr,
                            dali_data_type_t data_type, const int64_t *shapes,
                            int64_t sample_dim, const char *layout_str, int is_pinned);
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
 * @brief Return the shape of the output tensor
 * stored at position `n` in the pipeline.
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
 * @brief Copy the output tensor list stored
 * at position `n` in the pipeline.
 * dst_type (0 - CPU, 1 - GPU)
 * @remarks Tensor list doesn't need to be dense
 *
 * If you call this function with non_blocking != 0, make sure to
 * synchronize with the provided stream before reading the data.
 * If non_blocking == 0, function will do it for you
 */
DLL_PUBLIC void
daliCopyTensorListNTo(daliPipelineHandle *pipe_handle, void *dst, int n, device_type_t dst_type,
                      cudaStream_t stream, int non_blocking);

/**
 * @brief Returns number of DALI pipeline outputs
 */
DLL_PUBLIC unsigned daliGetNumOutput(daliPipelineHandle *pipe_handle);

/**
 * @brief Copy the output tensor stored
 * at position `n` in the pipeline.
 * dst_type (0 - CPU, 1 - GPU)
 * @remarks If the output is tensor list then it need to be dense
 *
 * If you call this function with non_blocking != 0, make sure to
 * synchronize on provided stream before reading the data.
 * If non_blocking == 0, function will do it for you
 */
DLL_PUBLIC void
daliCopyTensorNTo(daliPipelineHandle *pipe_handle, void *dst, int n, device_type_t dst_type,
                  cudaStream_t stream, int non_blocking);

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
