// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
