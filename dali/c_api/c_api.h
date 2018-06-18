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

// Trick to bypass gcc4.9 old ABI name mangling used by TF
extern "C" {
  struct PipelineHandle {
    void* pipe;
    void* ws;
  };

  void CreatePipeline(PipelineHandle* pipe_handle,
      const char *serialized_pipeline,
      int length,
      int batch_size,
      int num_threads,
      int device_id);
  void Run(PipelineHandle* pipe_handle);
  void Output(PipelineHandle* pipe_handle);
  void* TensorAt(PipelineHandle* pipe_handle, int n);
  int64_t* ShapeAt(PipelineHandle* pipe_handle, int n);
  void CopyTensorNTo(PipelineHandle* pipe_handle, void* dst, int n);
  void DeletePipeline(PipelineHandle* pipe_handle);
}

#endif  // DALI_C_API_C_API_H_
