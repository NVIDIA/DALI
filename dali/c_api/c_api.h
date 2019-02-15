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

#ifndef DALI_C_API_C_API_H_
#define DALI_C_API_C_API_H_

#include <inttypes.h>

#include "dali/api_helper.h"

// Trick to bypass gcc4.9 old ABI name mangling used by TF
extern "C" {
  struct daliPipelineHandle {
    void* pipe;
    void* ws;
  };

  enum device_type_t {
    CPU = 0,
    GPU = 1
  };

  /**
   * @brief Create DALI pipeline. Setting batch_size,
   * num_threads or device_id here overrides
   * values stored in the serialized pipeline.
   */
  DLL_PUBLIC void daliCreatePipeline(daliPipelineHandle* pipe_handle,
      const char *serialized_pipeline,
      int length,
      int batch_size = -1,
      int num_threads = -1,
      int device_id = -1,
      int prefetch_queue_depth = 2);

  /**
   * @brief Start the execution of the pipeline.
   */
  DLL_PUBLIC void daliRun(daliPipelineHandle* pipe_handle);

  /**
   * @brief Wait till the output of the pipeline is ready.
   * Releases previously returned buffers.
   */
  DLL_PUBLIC void daliOutput(daliPipelineHandle* pipe_handle);

  /**
   * @brief Wait till the output of the pipeline is ready.
   * Doesn't release previously returned buffers.
   */
  DLL_PUBLIC void daliShareOutput(daliPipelineHandle* pipe_handle);

  /**
   * @brief Releases buffer returned by last daliOutput call.
   */
  DLL_PUBLIC void daliOutputRelease(daliPipelineHandle* pipe_handle);

  /**
   * @brief Return the shape of the output tensor
   * stored at position `n` in the pipeline.
   * This function may only be called after
   * calling Output function.
   * @remarks Caller is responsible to 'free' the memory returned
   */
  DLL_PUBLIC int64_t* daliShapeAt(daliPipelineHandle* pipe_handle, int n);

  /**
   * @brief Return the shape of the 'k' output tensor from tensor list
   * stored at position `n` in the pipeline.
   * This function may only be called after
   * calling Output function.
   * @remarks Caller is responsible to 'free' the memory returned
   */
  DLL_PUBLIC int64_t* daliShapeAtSample(daliPipelineHandle* pipe_handle, int n, int k);

  /**
   * @brief Return the number of tensors in the tensor list
   * stored at position `n` in the pipeline.
   */
  DLL_PUBLIC size_t daliNumTensors(daliPipelineHandle* pipe_handle, int n);

  /**
   * @brief Return the number of all elements in the tensor list
   * stored at position `n` in the pipeline.
   */
  DLL_PUBLIC size_t daliNumElements(daliPipelineHandle* pipe_handle, int n);

  /**
   * @brief Return the size of the tensor list
   * stored at position `n` in the pipeline.
   */
  DLL_PUBLIC size_t daliTensorSize(daliPipelineHandle* pipe_handle, int n);

  /**
   * @brief Return maximum number of dimensions from all tensors
   * from the tensor list stored at position `n` in the pipeline.
   */
  DLL_PUBLIC size_t daliMaxDimTensors(daliPipelineHandle* pipe_handle, int n);

  /**
   * @brief Copy the output tensor list stored
   * at position `n` in the pipeline.
   * dst_type (0 - CPU, 1 - GPU)
   * @remarks Tensor list doesn't need to be dense
   */
  DLL_PUBLIC void daliCopyTensorListNTo(daliPipelineHandle* pipe_handle, void* dst, int n,
                                          device_type_t dst_type);

  /**
   * @brief Returns number of DALI pipeline outputs
   */
  DLL_PUBLIC unsigned daliGetNumOutput(daliPipelineHandle* pipe_handle);
  /**
   * @brief Copy the output tensor stored
   * at position `n` in the pipeline.
   * dst_type (0 - CPU, 1 - GPU)
   * @remarks If the output is tensor list then it need to be dense
   */
  DLL_PUBLIC void daliCopyTensorNTo(daliPipelineHandle* pipe_handle, void* dst, int n,
                                          device_type_t dst_type);

  /**
   * @brief Delete the pipeline object.
   */
  DLL_PUBLIC void daliDeletePipeline(daliPipelineHandle* pipe_handle);

  /**
   * @brief Load plugin library
   */
  DLL_PUBLIC void daliLoadLibrary(const char* lib_path);
}

#endif  // DALI_C_API_C_API_H_
