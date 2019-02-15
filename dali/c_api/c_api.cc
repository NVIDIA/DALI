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
#include <algorithm>

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

template <typename T>
static int64_t* daliShapeAtHelper(dali::DeviceWorkspace* ws, int n, int k) {
  int64_t* c_shape = nullptr;
  std::vector<dali::Index> shape;
  const auto &out_tensor_list = ws->Output<T>(n);
  if (k >= 0) {
    shape = out_tensor_list.tensor_shape(k);
  } else {
    shape = out_tensor_list.tensor_shape(0);
    shape.insert(shape.begin(), out_tensor_list.ntensor());
  }

  c_shape = static_cast<int64_t*>(malloc(sizeof(int64_t) * (shape.size() + 1)));
  c_shape[shape.size()] = 0;
  memcpy(c_shape, &shape[0], shape.size() * sizeof(int64_t));
  return c_shape;
}

static int64_t* daliShapeAtTypedHelper(daliPipelineHandle* pipe_handle, int n, int k) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    return daliShapeAtHelper<dali::CPUBackend>(ws, n, k);
  } else {
    return daliShapeAtHelper<dali::GPUBackend>(ws, n, k);
  }
}

int64_t* daliShapeAtSample(daliPipelineHandle* pipe_handle, int n, int k) {
  return daliShapeAtTypedHelper(pipe_handle, n, k);
}

int64_t* daliShapeAt(daliPipelineHandle* pipe_handle, int n) {
  return daliShapeAtTypedHelper(pipe_handle, n, -1);
}

template <typename T>
static size_t daliNumTensorsHelper(dali::DeviceWorkspace* ws, int n) {
  return ws->Output<T>(n).ntensor();
}

size_t daliNumTensors(daliPipelineHandle* pipe_handle, int n) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    return daliNumTensorsHelper<dali::CPUBackend>(ws, n);
  } else {
    return daliNumTensorsHelper<dali::GPUBackend>(ws, n);
  }
}

template <typename T>
static size_t daliNumElementsHelper(dali::DeviceWorkspace* ws, int n) {
  return ws->Output<T>(n).GetElementsNumber();
}

size_t daliNumElements(daliPipelineHandle* pipe_handle, int n) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    return daliNumElementsHelper<dali::CPUBackend>(ws, n);
  } else {
    return daliNumElementsHelper<dali::GPUBackend>(ws, n);
  }
}

template <typename T>
static size_t daliTensorSizeHelper(dali::DeviceWorkspace* ws, int n) {
  return ws->Output<T>(n).nbytes();
}

size_t daliTensorSize(daliPipelineHandle* pipe_handle, int n) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    return daliTensorSizeHelper<dali::CPUBackend>(ws, n);
  } else {
    return daliTensorSizeHelper<dali::GPUBackend>(ws, n);
  }
}

template <typename T>
static size_t daliMaxDimTensorsHelper(dali::DeviceWorkspace* ws, int n) {
  const auto &out_tensor_list = ws->Output<T>(n);
  size_t tensors_num = out_tensor_list.ntensor();
  size_t max_num_dim = 0;
  for (size_t i = 0; i < tensors_num; ++i) {
    size_t num_dim = out_tensor_list.tensor_shape(i).size();
    // squeeze last dimension
    if (out_tensor_list.tensor_shape(i).back() == 1) {
      --num_dim;
    }
    max_num_dim = std::max(max_num_dim, num_dim);
  }
  return max_num_dim;
}

size_t daliMaxDimTensors(daliPipelineHandle* pipe_handle, int n) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    return daliMaxDimTensorsHelper<dali::CPUBackend>(ws, n);
  } else {
    return daliMaxDimTensorsHelper<dali::GPUBackend>(ws, n);
  }
}

unsigned daliGetNumOutput(daliPipelineHandle* pipe_handle) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  return ws->NumOutput();
}

template <typename T>
static void daliCopyTensorListNToHelper(dali::DeviceWorkspace* ws, void* dst, int n,
                                        device_type_t dst_type) {
  dali::CopyToExternalTensor(&(ws->Output<T>(n)), dst, (dali::device_type_t)dst_type);
}

void daliCopyTensorListNTo(daliPipelineHandle* pipe_handle, void* dst, int n,
                           device_type_t dst_type) {
  dali::TimeRange tr("daliCopyTensorNTo", dali::TimeRange::kGreen);
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    daliCopyTensorListNToHelper<dali::CPUBackend>(ws, dst, n, dst_type);
  } else {
    daliCopyTensorListNToHelper<dali::GPUBackend>(ws, dst, n, dst_type);
  }
}

template <typename T>
static void daliCopyTensorNToHelper(dali::DeviceWorkspace* ws, void* dst, int n,
                                    device_type_t dst_type) {
  dali::Tensor<T> t;
  t.ShareData(&(ws->Output<T>(n)));
  dali::CopyToExternalTensor(t, dst, (dali::device_type_t)dst_type);
}

void daliCopyTensorNTo(daliPipelineHandle* pipe_handle, void* dst, int n, device_type_t dst_type) {
  dali::TimeRange tr("daliCopyTensorNTo", dali::TimeRange::kGreen);
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    daliCopyTensorNToHelper<dali::CPUBackend>(ws, dst, n, dst_type);
  } else {
    daliCopyTensorNToHelper<dali::GPUBackend>(ws, dst, n, dst_type);
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
