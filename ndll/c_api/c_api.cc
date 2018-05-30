// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include <string>
#include <vector>

#include "ndll/c_api/c_api.h"
#include "ndll/pipeline/pipeline.h"
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
                              -1,
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
  ndll::DeviceWorkspace* ws = reinterpret_cast<ndll::DeviceWorkspace*>(pipe_handle->ws);
  pipeline->Outputs(ws);
}

void* TensorAt(PipelineHandle* pipe_handle, int n) {
  ndll::DeviceWorkspace* ws = reinterpret_cast<ndll::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<ndll::CPUBackend>(n)) {
    ndll::Tensor<ndll::CPUBackend> *t = new ndll::Tensor<ndll::CPUBackend>();
    t->ShareData(ws->Output<ndll::CPUBackend>(n));
    return t;
  } else {
    ndll::Tensor<ndll::GPUBackend> *t = new ndll::Tensor<ndll::GPUBackend>();
    t->ShareData(ws->Output<ndll::GPUBackend>(n));
    return t;
  }
}

int64_t* ShapeAt(PipelineHandle* pipe_handle, int n) {
  ndll::DeviceWorkspace* ws = reinterpret_cast<ndll::DeviceWorkspace*>(pipe_handle->ws);
  int64_t* c_shape = nullptr;
  if (ws->OutputIsType<ndll::CPUBackend>(n)) {
    ndll::Tensor<ndll::CPUBackend> t;
    t.ShareData(ws->Output<ndll::CPUBackend>(n));

    std::vector<ndll::Index> shape = t.shape();
    c_shape = new int64_t[shape.size() + 1];
    c_shape[shape.size()] = 0;
    memcpy(c_shape, &shape[0], shape.size() * sizeof (int64_t));
  } else {
    ndll::Tensor<ndll::GPUBackend> t;
    t.ShareData(ws->Output<ndll::GPUBackend>(n));

    std::vector<ndll::Index> shape = t.shape();
    c_shape = new int64_t[shape.size() + 1];
    c_shape[shape.size()] = 0;
    memcpy(c_shape, &shape[0], shape.size() * sizeof (int64_t));
  }
  return c_shape;
}

void CopyTensorNTo(PipelineHandle* pipe_handle, void* dst, int n) {
  ndll::DeviceWorkspace* ws = reinterpret_cast<ndll::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<ndll::CPUBackend>(n)) {
    ndll::Tensor<ndll::CPUBackend> t;
    t.ShareData(ws->Output<ndll::CPUBackend>(n));
    ndll::CopyToExternalTensor(t, dst);
  } else {
    ndll::Tensor<ndll::GPUBackend> t;
    t.ShareData(ws->Output<ndll::GPUBackend>(n));
    ndll::CopyToExternalTensor(t, dst);
  }
}

void DeletePipeline(PipelineHandle* pipe_handle) {
  ndll::Pipeline* pipeline = reinterpret_cast<ndll::Pipeline*>(pipe_handle->pipe);
  ndll::DeviceWorkspace* ws = reinterpret_cast<ndll::DeviceWorkspace*>(pipe_handle->ws);
  NDLL_ENFORCE(pipeline != nullptr && ws != nullptr, "Pipeline already deleted");
  delete ws;
  delete pipeline;
  pipe_handle->ws = nullptr;
  pipe_handle->pipe = nullptr;
}

