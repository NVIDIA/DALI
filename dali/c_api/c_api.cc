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


#include <string>
#include <vector>

#include "dali/c_api/c_api.h"
#include "dali/pipeline/pipeline.h"
#include "dali/plugin/copy.h"
#include "dali/plugin/plugin_manager.h"

void daliCreatePipeline(daliPipelineHandle* pipe_handle,
    const char *serialized_pipeline,
    int length,
    int batch_size,
    int num_threads,
    int device_id,
    int prefetch_queue_depth) {
  dali::Pipeline* pipe = new dali::Pipeline(
                              std::string(serialized_pipeline, length),
                              batch_size,
                              num_threads,
                              device_id,
                              true,
                              prefetch_queue_depth,
                              true);
  pipe->Build();
  pipe_handle->pipe = reinterpret_cast<void*>(pipe);
  pipe_handle->ws = new dali::DeviceWorkspace();
}

void daliRun(daliPipelineHandle* pipe_handle) {
  dali::Pipeline* pipeline = reinterpret_cast<dali::Pipeline*>(pipe_handle->pipe);
  pipeline->RunCPU();
  pipeline->RunGPU();
}

void daliOutput(daliPipelineHandle* pipe_handle) {
  dali::Pipeline* pipeline = reinterpret_cast<dali::Pipeline*>(pipe_handle->pipe);
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  pipeline->Outputs(ws);
}

void daliShareOutput(daliPipelineHandle* pipe_handle) {
  dali::Pipeline* pipeline = reinterpret_cast<dali::Pipeline*>(pipe_handle->pipe);
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  pipeline->ShareOutputs(ws);
}

void daliOutputRelease(daliPipelineHandle* pipe_handle) {
  dali::Pipeline* pipeline = reinterpret_cast<dali::Pipeline*>(pipe_handle->pipe);
  pipeline->ReleaseOutputs();
}

int64_t* daliShapeAt(daliPipelineHandle* pipe_handle, int n) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  int64_t* c_shape = nullptr;
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    dali::Tensor<dali::CPUBackend> t;
    t.ShareData(ws->Output<dali::CPUBackend>(n));

    std::vector<dali::Index> shape = t.shape();
    c_shape = static_cast<int64_t*>(
      malloc(sizeof(int64_t) * (shape.size() + 1)));
    c_shape[shape.size()] = 0;
    memcpy(c_shape, &shape[0], shape.size() * sizeof(int64_t));
  } else {
    dali::Tensor<dali::GPUBackend> t;
    t.ShareData(ws->Output<dali::GPUBackend>(n));

    std::vector<dali::Index> shape = t.shape();
    c_shape = static_cast<int64_t*>(
      malloc(sizeof(int64_t) * (shape.size() + 1)));
    c_shape[shape.size()] = 0;
    memcpy(c_shape, &shape[0], shape.size() * sizeof(int64_t));
  }
  return c_shape;
}

unsigned daliGetNumOutput(daliPipelineHandle* pipe_handle) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  return ws->NumOutput();
}

void daliCopyTensorNTo(daliPipelineHandle* pipe_handle, void* dst, int n, device_type_t dst_type) {
  dali::TimeRange tr("daliCopyTensorNTo", dali::TimeRange::kGreen);
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    dali::Tensor<dali::CPUBackend> t;
    t.ShareData(ws->Output<dali::CPUBackend>(n));
    dali::CopyToExternalTensor(t, dst, (dali::device_type_t)dst_type);
  } else {
    dali::Tensor<dali::GPUBackend> t;
    t.ShareData(ws->Output<dali::GPUBackend>(n));
    dali::CopyToExternalTensor(t, dst, (dali::device_type_t)dst_type);
  }
}

void daliDeletePipeline(daliPipelineHandle* pipe_handle) {
  dali::Pipeline* pipeline = reinterpret_cast<dali::Pipeline*>(pipe_handle->pipe);
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  DALI_ENFORCE(pipeline != nullptr && ws != nullptr, "Pipeline already deleted");
  delete ws;
  delete pipeline;
  pipe_handle->ws = nullptr;
  pipe_handle->pipe = nullptr;
}

void daliLoadLibrary(const char* lib_path) {
    dali::PluginManager::LoadLibrary(lib_path);
}
