// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/common.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/pipeline/tfpipeline.h"
#include "ndll/plugin/copy.h"

void CreatePipeline(PipelineHandle* pipe_handle,
    const char *serialized_pipeline,
    int length,
    int batch_size,
    int num_threads,
    int device_id) {
  ndll::Pipeline* pipe = new ndll::Pipeline(
                              std::string(serialized_pipeline, length),
                              batch_size,
                              num_threads,
                              device_id,
                              true,
                              true);
  pipe->Build();
  pipe_handle->pipe = reinterpret_cast<void*>(pipe);
  pipe_handle->ws = new ndll::DeviceWorkspace();
}

void Run(PipelineHandle* pipe_handle) {
  ndll::Pipeline* pipeline = reinterpret_cast<ndll::Pipeline*>(pipe_handle->pipe);
  pipeline->RunCPU();
  pipeline->RunGPU();
}

void Output(PipelineHandle* pipe_handle) {
  ndll::Pipeline* pipeline = reinterpret_cast<ndll::Pipeline*>(pipe_handle->pipe);
  pipeline->Outputs(pipe_handle->ws);
}

void* TensorAt(PipelineHandle* pipe_handle, int n) {
  if (pipe_handle->ws->OutputIsType<ndll::CPUBackend>(n)) {
    ndll::Tensor<ndll::CPUBackend> *t = new ndll::Tensor<ndll::CPUBackend>();
    t->ShareData(pipe_handle->ws->Output<ndll::CPUBackend>(n));
    return t;
  } else {
    ndll::Tensor<ndll::GPUBackend> *t = new ndll::Tensor<ndll::GPUBackend>();
    t->ShareData(pipe_handle->ws->Output<ndll::GPUBackend>(n));
    return t;
  }
}

std::vector<ndll::Index> ShapeAt(PipelineHandle* pipe_handle, int n) {
  if (pipe_handle->ws->OutputIsType<ndll::CPUBackend>(n)) {
    ndll::Tensor<ndll::CPUBackend> t;
    t.ShareData(pipe_handle->ws->Output<ndll::CPUBackend>(n));
    return t.shape();
  } else {
    ndll::Tensor<ndll::GPUBackend> t;
    t.ShareData(pipe_handle->ws->Output<ndll::GPUBackend>(n));
    return t.shape();
  }
}

void CopyTensorNTo(PipelineHandle* pipe_handle, void* dst, int n) {
  if (pipe_handle->ws->OutputIsType<ndll::CPUBackend>(n)) {
    ndll::Tensor<ndll::CPUBackend> t;
    t.ShareData(pipe_handle->ws->Output<ndll::CPUBackend>(n));
    ndll::CopyToExternalTensor(t, dst);
  } else {
    ndll::Tensor<ndll::GPUBackend> t;
    t.ShareData(pipe_handle->ws->Output<ndll::GPUBackend>(n));
    ndll::CopyToExternalTensor(t, dst);
  }
}

void DeletePipeline(PipelineHandle* pipe_handle) {
  ndll::Pipeline* pipeline = reinterpret_cast<ndll::Pipeline*>(pipe_handle->pipe);
  ndll::DeviceWorkspace* ws = pipe_handle->ws;
  delete ws;
  delete pipeline;
}

