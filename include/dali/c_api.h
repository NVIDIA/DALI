// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
} daliPipelineHandle;

typedef enum {
  CPU = 0,
  GPU = 1
} device_type_t;

typedef enum {
  DALI_NO_TYPE = -1,
  DALI_UINT8 = 0,
  DALI_UINT16 = 1,
  DALI_UINT32 = 2,
  DALI_UINT64 = 3,
  DALI_INT8 = 4,
  DALI_INT16 = 5,
  DALI_INT32 = 6,
  DALI_INT64 = 7,
  DALI_FLOAT16 = 8,
  DALI_FLOAT = 9,
  DALI_FLOAT64 = 10,
  DALI_BOOL = 11
} dali_data_type_t;

typedef enum {
  DALI_SUCCESS = 0,
  DALI_GENERAL_ERROR=1,
  DALI_UNKNOWN_DEVICE = 2,
} dali_error_t;

/**
 * @brief Create DALI pipeline. Setting batch_size,
 * num_threads or device_id here overrides
 * values stored in the serialized pipeline.
 * When separated_execution is false, prefetch_queue_depth is considered,
 * gpu_prefetch_queue_depth and cpu_prefetch_queue_depth are ignored.
 * When separated_execution is true, cpu_prefetch_queue_depth and
 * gpu_prefetch_queue_depth are considered and prefetch_queue_depth is ignored.
 */
DLL_PUBLIC void daliCreatePipeline(daliPipelineHandle *pipe_handle,
                                   const char *serialized_pipeline,
                                   int length,
                                   int batch_size,
                                   int num_threads,
                                   int device_id,
                                   bool separated_execution,
                                   int prefetch_queue_depth,
                                   int cpu_prefetch_queue_depth,
                                   int gpu_prefetch_queue_depth);

/**
 * @brief Feed the data to ExternalSource as contiguous memory.
 *
 * An alternative function exists to specify also a CUDA stream to use
 * when copying the data onto GPU. If used with `device == CPU`, this value will be ignored
 *
 * @param pipe_handle Pointer to pipeline handle
 * @param name Pointer to a null-terminated byte string with the name of the External Source
 *             to be fed
 * @param device Device of the supplied memory. Only CPU is supported.
 * @param data_ptr Pointer to contiguous buffer containing all samples
 * @param data_type Type of the provided data
 * @param shapes Pointer to an array containing shapes of all samples concatenated one after
 *              another. Should contain batch_size * sample_dim elements.
 * @param sample_dim The dimensionality of a single sample.
 * @param layout_str Optional layout provided as a pointer to null-terminated byte string.
 *                   Can be set to NULL.
 * @param stream CUDA stream to use when copying the data onto GPU. If used with `device == CPU`,
 *               this value will be ignored
 */
DLL_PUBLIC void daliSetExternalInput(daliPipelineHandle *pipe_handle, const char *name,
                                     device_type_t device, const void *data_ptr,
                                     dali_data_type_t data_type, const int64_t *shapes,
                                     int sample_dim, const char *layout_str);

DLL_PUBLIC void daliSetExternalInputStream(daliPipelineHandle *pipe_handle, const char *name,
                                           device_type_t device, const void *data_ptr,
                                           dali_data_type_t data_type, const int64_t *shapes,
                                           int sample_dim, const char *layout_str,
                                           cudaStream_t stream);

/**
 * @brief Feed the data to ExternalSource as a set of separate buffers.
 *
 * An alternative function exists to specify also a CUDA stream to use
 * when copying the data onto GPU. If used with `device == CPU`, this value will be ignored
 *
 * @param pipe_handle Pointer to pipeline handle
 * @param name Pointer to a null-terminated byte string with the name of the External Source
 *             to be fed
 * @param device Device of the supplied memory. Only CPU is supported.
 * @param data_ptr Pointer to an array containing batch_size pointers to separate Tensors.
 * @param data_type Type of the provided data
 * @param shapes Pointer to an array containing shapes of all samples concatenated one after
 *              another. Should contain batch_size * sample_dim elements.
 * @param sample_dim The dimensionality of a single sample.
 * @param layout_str Optional layout provided as a pointer to null-terminated byte string.
 *                   Can be set to NULL.
 * @param stream CUDA stream to use when copying the data onto GPU. If used with `device == CPU`,
 *               this value will be ignored
 */
DLL_PUBLIC void daliSetExternalInputTensors(daliPipelineHandle *pipe_handle, const char *name,
                                            device_type_t device, const void *const *data_ptr,
                                            dali_data_type_t data_type, const int64_t *shapes,
                                            int64_t sample_dim, const char *layout_str);

DLL_PUBLIC void daliSetExternalInputTensorsStream(daliPipelineHandle *pipe_handle, const char *name,
                                                  device_type_t device, const void *const *data_ptr,
                                                  dali_data_type_t data_type, const int64_t *shapes,
                                                  int64_t sample_dim, const char *layout_str,
                                                  cudaStream_t stream);

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
 */
DLL_PUBLIC void daliCopyTensorListNTo(daliPipelineHandle *pipe_handle, void *dst, int n,
                                      device_type_t dst_type, cudaStream_t stream,
                                      bool non_blocking);

/**
 * @brief Returns number of DALI pipeline outputs
 */
DLL_PUBLIC unsigned daliGetNumOutput(daliPipelineHandle *pipe_handle);
/**
 * @brief Copy the output tensor stored
 * at position `n` in the pipeline.
 * dst_type (0 - CPU, 1 - GPU)
 * @remarks If the output is tensor list then it need to be dense
 */
DLL_PUBLIC void daliCopyTensorNTo(daliPipelineHandle *pipe_handle, void *dst, int n,
                                  device_type_t dst_type, cudaStream_t stream,
                                  bool non_blocking);

/**
 * @brief Delete the pipeline object.
 */
DLL_PUBLIC void daliDeletePipeline(daliPipelineHandle *pipe_handle);

/**
 * @brief Load plugin library
 */
DLL_PUBLIC void daliLoadLibrary(const char *lib_path);

#ifdef __cplusplus
}
#endif

#endif  // DALI_C_API_H_
