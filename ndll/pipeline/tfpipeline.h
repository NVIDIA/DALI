// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_TFPIPELINE_H_
#define NDLL_PIPELINE_TFPIPELINE_H_

#include <vector>
#include <string>

#include "ndll/common.h"
#include "ndll/pipeline/pipeline.h"

// Trick to bypass gcc4.9 old ABI name mangling used by TF
extern "C" {
  struct PipelineHandle {
    void* pipe;
    ndll::DeviceWorkspace* ws;
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
  std::vector<ndll::Index> ShapeAt(PipelineHandle* pipe_handle, int n);
  void CopyTensorNTo(PipelineHandle* pipe_handle, void* dst, int n);
  void DeletePipeline(PipelineHandle* pipe_handle);
}

#endif  // NDLL_PIPELINE_TFPIPELINE_H_
