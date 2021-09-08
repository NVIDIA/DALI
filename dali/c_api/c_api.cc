// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/c_api.h"  // NOLINT [build/include]

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "dali/core/common.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/format.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/init.h"

#include "dali/pipeline/pipeline.h"
#include "dali/plugin/plugin_manager.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/copy_to_external.h"

namespace {

bool dali_initialized = false;

/**
 * Maps operator name to the batch size set prior to daliSetExternal... call.
 * Typically, this operator will be BatchSizeProvider.
 * Negative values denote max batch size (default state).
 * Typical usage:
 * auto batch_sizes_map = reinterpret_cast<batch_size_map_t*>(handle->batch_sizes_map);
 */
using batch_size_map_t = std::unordered_map<std::string /* op_name */, int /* batch_size */>;


int PopCurrBatchSize(batch_size_map_t *batch_size_map, int max_batch_size,
                     const std::string &op_name) {
  auto it = batch_size_map->find(op_name);
  auto exists = it != batch_size_map->end();
  auto ret = !exists || it->second < 0 ? max_batch_size : it->second;
  if (exists) {
    it->second = -1;
  }
  return ret;
}

/**
 * @brief Extract ExtSrcNoCopyMode based on the DALI_ext_force_copy and DALI_ext_force_no_copy
 *
 * @param flags Flags typically specified in daliSetExternalInput* functions.
 */
dali::ExtSrcNoCopyMode GetExternalSourceCopyMode(unsigned int flags) {
  dali::ExtSrcNoCopyMode no_copy_mode = dali::ExtSrcNoCopyMode::DEFAULT;
  DALI_ENFORCE(!((flags & DALI_ext_force_copy) && (flags & DALI_ext_force_no_copy)),
               "External Source cannot be forced to use DALI_ext_force_copy and "
               "DALI_ext_force_no_copy at the same time.");
  if (flags & DALI_ext_force_copy) {
    no_copy_mode = dali::ExtSrcNoCopyMode::FORCE_COPY;
  } else if (flags & DALI_ext_force_no_copy) {
    no_copy_mode = dali::ExtSrcNoCopyMode::FORCE_NO_COPY;
  }
  return no_copy_mode;
}

template <typename Backend>
void SetExternalInput(daliPipelineHandle *pipe_handle, const char *name, const void *data_ptr,
                      dali_data_type_t data_type, const int64_t *shapes, int sample_dim,
                      const char *layout_str, cudaStream_t stream = 0, unsigned int flags = 0) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  auto bs_map = reinterpret_cast<batch_size_map_t *>(pipe_handle->batch_sizes_map);
  auto curr_batch_size = PopCurrBatchSize(bs_map, pipeline->max_batch_size(), name);
  std::vector<int64_t> shapes_tmp(shapes, shapes + sample_dim * curr_batch_size);
  dali::TensorListShape<> tl_shape(std::move(shapes_tmp), curr_batch_size, sample_dim);
  dali::TensorLayout layout{};
  if (layout_str != nullptr) {
    layout = dali::TensorLayout(layout_str);
  }
  dali::TensorList<Backend> data;
  const auto &type_info = dali::TypeTable::GetTypeInfo(static_cast<dali::DALIDataType>(data_type));
  auto elem_sizeof = type_info.size();
  // We cast away the const from data_ptr, as there is no other way of passing it to the
  // TensorList, as we must also set the shape and type metadata.
  // It is passed further as const TensorList, so it's data cannot be modified.
  data.set_pinned(flags & DALI_ext_pinned);
  data.ShareData(const_cast<void *>(data_ptr), tl_shape.num_elements() * elem_sizeof);
  data.Resize(tl_shape, type_info);
  data.SetLayout(layout);
  pipeline->SetExternalInput(name, data, stream,
                             flags & DALI_ext_force_sync,
                             flags & DALI_use_copy_kernel,
                             GetExternalSourceCopyMode(flags));
}


template<typename Backend>
void SetExternalInputTensors(daliPipelineHandle *pipe_handle, const char *name,
                             const void *const *data_ptr, dali_data_type_t data_type,
                             const int64_t *shapes, int64_t sample_dim, const char *layout_str,
                             cudaStream_t stream = 0, unsigned int flags = 0) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  auto bs_map = reinterpret_cast<batch_size_map_t *>(pipe_handle->batch_sizes_map);
  auto curr_batch_size = PopCurrBatchSize(bs_map, pipeline->max_batch_size(), name);
  std::vector<int64_t> shapes_tmp(shapes, shapes + sample_dim * curr_batch_size);
  dali::TensorListShape<> tl_shape(std::move(shapes_tmp), curr_batch_size, sample_dim);
  dali::TensorLayout layout{};
  if (layout_str != nullptr) {
    layout = dali::TensorLayout(layout_str);
  }
  dali::TensorVector<Backend> data(curr_batch_size);
  const auto &type_info = dali::TypeTable::GetTypeInfo(static_cast<dali::DALIDataType>(data_type));
  auto elem_sizeof = type_info.size();
  for (int i = 0; i < curr_batch_size; i++) {
    // We cast away the const from data_ptr, as there is no other way of passing it to the
    // Tensor as we must also set the shape and type metadata.
    // The vector that we pass to pipeline is const.
    data[i].set_pinned(flags & DALI_ext_pinned);
    data[i].ShareData(const_cast<void *>(data_ptr[i]), tl_shape[i].num_elements() * elem_sizeof);
    data[i].Resize(tl_shape[i], type_info);
    data[i].SetLayout(layout);
  }
  pipeline->SetExternalInput(name, data, stream,
                             flags & DALI_ext_force_sync,
                             flags & DALI_use_copy_kernel,
                             GetExternalSourceCopyMode(flags));
}

inline dali::mm::memory_kind_id GetMemKind(device_type_t device_type, bool is_pinned) {
  return device_type == device_type_t::GPU
        ? dali::mm::memory_kind_id::device
        : (is_pinned ? dali::mm::memory_kind_id::pinned : dali::mm::memory_kind_id::host);
}

}  // namespace


void daliInitialize() {
  static std::once_flag init_flag;
  auto init = [&] {
      dali::DALIInit(dali::OpSpec("CPUAllocator"),
                     dali::OpSpec("PinnedCPUAllocator"),
                     dali::OpSpec("GPUAllocator"));
      dali_initialized = true;
  };
  std::call_once(init_flag, init);
}

void daliCreatePipeline(daliPipelineHandle *pipe_handle, const char *serialized_pipeline,
                        int length, int max_batch_size, int num_threads, int device_id,
                        int separated_execution, int prefetch_queue_depth,
                        int cpu_prefetch_queue_depth, int gpu_prefetch_queue_depth,
                        int enable_memory_stats) {
  bool se = separated_execution != 0;
  auto pipeline =
      std::make_unique<dali::Pipeline>(std::string(serialized_pipeline, length), max_batch_size,
                                       num_threads, device_id, true, prefetch_queue_depth, true);
  pipeline->SetExecutionTypes(true, se, true);
  if (se) {
    pipeline->SetQueueSizes(cpu_prefetch_queue_depth, gpu_prefetch_queue_depth);
  }
  pipeline->EnableExecutorMemoryStats(enable_memory_stats);
  pipeline->Build();
  auto ws = std::make_unique<dali::DeviceWorkspace>();
  dali::CUDAStream stream;
  if (pipeline->device_id() >= 0) {
    stream = dali::CUDAStream::Create(true);
  }
  pipe_handle->ws = ws.release();
  pipe_handle->copy_stream = stream.release();
  pipe_handle->pipe = pipeline.release();

  auto bs_map = std::make_unique<batch_size_map_t>();
  pipe_handle->batch_sizes_map = bs_map.release();
}


void daliDeserializeDefault(daliPipelineHandle *pipe_handle, const char *serialized_pipeline,
                            int length) {
  auto pipeline = std::make_unique<dali::Pipeline>(std::string(serialized_pipeline, length));
  pipeline->Build();
  dali::CUDAStream stream;
  if (pipeline->device_id() >= 0) {
    stream = dali::CUDAStream::Create(true);
  }
  auto ws = std::make_unique<dali::DeviceWorkspace>();
  pipe_handle->ws = ws.release();
  pipe_handle->copy_stream = stream.release();
  pipe_handle->pipe = pipeline.release();
}


void daliPrefetchUniform(daliPipelineHandle *pipe_handle, int queue_depth) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  for (int i = 0; i < queue_depth; ++i) {
    pipeline->RunCPU();
    pipeline->RunGPU();
  }
}


void daliPrefetchSeparate(daliPipelineHandle *pipe_handle,
                          int cpu_queue_depth, int gpu_queue_depth) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  for (int i = 0; i < gpu_queue_depth; ++i) {
    pipeline->RunCPU();
    pipeline->RunGPU();
  }
  for (int i = 0; i < cpu_queue_depth; ++i) {
    pipeline->RunCPU();
  }
}


void daliSetExternalInputBatchSize(daliPipelineHandle *pipe_handle, const char *name,
                                   int batch_size) {
  auto *bs_map = reinterpret_cast<batch_size_map_t *>(pipe_handle->batch_sizes_map);
  (*bs_map)[name] = batch_size;
}


void daliSetExternalInput(daliPipelineHandle *pipe_handle, const char *name, device_type_t device,
                          const void *data_ptr, dali_data_type_t data_type, const int64_t *shapes,
                          int sample_dim, const char *layout_str, unsigned int flags) {
  daliSetExternalInputAsync(pipe_handle, name, device, data_ptr, data_type, shapes, sample_dim,
                            layout_str, pipe_handle->copy_stream, flags | DALI_ext_force_sync);
}

void daliSetExternalInputAsync(daliPipelineHandle *pipe_handle, const char *name,
                               device_type_t device, const void *data_ptr,
                               dali_data_type_t data_type, const int64_t *shapes,
                               int sample_dim, const char *layout_str, cudaStream_t stream,
                               unsigned int flags) {
  switch (device) {
    case device_type_t::CPU:
      SetExternalInput<dali::CPUBackend>(pipe_handle, name, data_ptr, data_type, shapes, sample_dim,
                                         layout_str, stream, flags);
      return;
    case device_type_t::GPU:
      SetExternalInput<dali::GPUBackend>(pipe_handle, name, data_ptr, data_type, shapes, sample_dim,
                                         layout_str, stream, flags);
      return;
    default:
      DALI_FAIL(dali::make_string("Unknown device: ", device));
  }
}


void daliSetExternalInputTensors(daliPipelineHandle *pipe_handle, const char *name,
                                 device_type_t device, const void *const *data_ptr,
                                 dali_data_type_t data_type, const int64_t *shapes,
                                 int64_t sample_dim, const char *layout_str, unsigned int flags) {
  daliSetExternalInputTensorsAsync(pipe_handle, name, device, data_ptr, data_type, shapes,
                                        sample_dim, layout_str, pipe_handle->copy_stream,
                                        flags | DALI_ext_force_sync);
}


void daliSetExternalInputTensorsAsync(daliPipelineHandle *pipe_handle, const char *name,
                                      device_type_t device, const void *const *data_ptr,
                                      dali_data_type_t data_type, const int64_t *shapes,
                                      int64_t sample_dim, const char *layout_str,
                                      cudaStream_t stream, unsigned int flags) {
  switch (device) {
    case device_type_t::CPU:
      SetExternalInputTensors<dali::CPUBackend>(pipe_handle, name, data_ptr, data_type, shapes,
                                                sample_dim, layout_str, stream, flags);
      return;
    case device_type_t::GPU:
      SetExternalInputTensors<dali::GPUBackend>(pipe_handle, name, data_ptr, data_type, shapes,
                                                sample_dim, layout_str, stream, flags);
      return;
    default:
      DALI_FAIL(dali::make_string("Unknown device: ", device));
  }
}


void daliRun(daliPipelineHandle *pipe_handle) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  pipeline->RunCPU();
  pipeline->RunGPU();
}


void daliOutput(daliPipelineHandle *pipe_handle) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  dali::DeviceWorkspace *ws = reinterpret_cast<dali::DeviceWorkspace *>(pipe_handle->ws);
  pipeline->Outputs(ws);
}


void daliShareOutput(daliPipelineHandle *pipe_handle) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  dali::DeviceWorkspace *ws = reinterpret_cast<dali::DeviceWorkspace *>(pipe_handle->ws);
  pipeline->ShareOutputs(ws);
}


void daliOutputRelease(daliPipelineHandle *pipe_handle) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  pipeline->ReleaseOutputs();
}

int64_t daliOutputHasUniformShape(daliPipelineHandle* pipe_handle, int i) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(i)) {
    return is_uniform(ws->Output<dali::CPUBackend>(i).shape());
  } else {
    return is_uniform(ws->Output<dali::GPUBackend>(i).shape());
  }
}

template<typename T>
static int64_t *daliShapeAtHelper(dali::DeviceWorkspace *ws, int n, int k) {
  int64_t *c_shape = nullptr;
  std::vector<dali::Index> shape;
  const auto &out_tensor_list = ws->Output<T>(n);
  if (k >= 0) {
    auto shape_span = out_tensor_list.tensor_shape_span(k);
    shape = std::vector<dali::Index>(shape_span.begin(), shape_span.end());
  } else {
    auto shape_span = out_tensor_list.tensor_shape_span(0);
    shape = std::vector<dali::Index>(shape_span.begin(), shape_span.end());
    shape.insert(shape.begin(), out_tensor_list.ntensor());
  }

  c_shape = static_cast<int64_t*>(malloc(sizeof(int64_t) * (shape.size() + 1)));
  if (!c_shape) {
    return nullptr;
  }
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
static dali_data_type_t daliTypeAtHelper(dali::DeviceWorkspace* ws, int n) {
  const auto &out_tensor_list = ws->Output<T>(n);
  auto type_id = out_tensor_list.type().id();
  return static_cast<dali_data_type_t>(static_cast<int>(type_id));
}

dali_data_type_t daliTypeAt(daliPipelineHandle* pipe_handle, int n) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    return daliTypeAtHelper<dali::CPUBackend>(ws, n);
  } else {
    return daliTypeAtHelper<dali::GPUBackend>(ws, n);
  }
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
  int max_num_dim = 0;
  for (size_t i = 0; i < tensors_num; ++i) {
    auto shape = out_tensor_list.tensor_shape(i);
    int num_dim = shape.size();
    // squeeze last dimension
    if (shape[num_dim - 1] == 1) {
      --num_dim;
    }
    max_num_dim = std::max(max_num_dim, num_dim);
  }
  return static_cast<size_t>(max_num_dim);
}

size_t daliMaxDimTensors(daliPipelineHandle* pipe_handle, int n) {
  dali::DeviceWorkspace* ws = reinterpret_cast<dali::DeviceWorkspace*>(pipe_handle->ws);
  if (ws->OutputIsType<dali::CPUBackend>(n)) {
    return daliMaxDimTensorsHelper<dali::CPUBackend>(ws, n);
  } else {
    return daliMaxDimTensorsHelper<dali::GPUBackend>(ws, n);
  }
}

unsigned daliGetNumOutput(daliPipelineHandle *pipe_handle) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  return pipeline->num_outputs();
}

const char *daliGetOutputName(daliPipelineHandle *pipe_handle, int id) {
  auto *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  return pipeline->output_name(id).c_str();
}

device_type_t daliGetOutputDevice(daliPipelineHandle *pipe_handle, int id) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  if (pipeline->output_device(id) == "gpu") {
    return device_type_t::GPU;
  }
  return device_type_t::CPU;
}

void daliOutputCopy(daliPipelineHandle *pipe_handle, void *dst, int output_idx,
                    device_type_t dst_type, cudaStream_t stream, unsigned int flags) {
  dali::DomainTimeRange tr("[DALI][C API] daliOutputCopy", dali::DomainTimeRange::kGreen);

  bool is_pinned = flags & DALI_ext_pinned;
  bool sync = flags & DALI_ext_force_sync;
  bool use_copy_kernel = flags & DALI_use_copy_kernel;
  auto dst_mem_kind = GetMemKind(dst_type, is_pinned);

  dali::DeviceWorkspace *ws = reinterpret_cast<dali::DeviceWorkspace *>(pipe_handle->ws);
  assert(ws != nullptr);

  auto &type_info = dali::TypeTable::GetTypeInfo(dali::DALIDataType::DALI_UINT8);
  if (ws->OutputIsType<dali::CPUBackend>(output_idx)) {
    CopyToExternal(dst, dst_mem_kind, ws->OutputRef<dali::CPUBackend>(output_idx),
                   stream, use_copy_kernel);
  } else {
    CopyToExternal(dst, dst_mem_kind, ws->OutputRef<dali::GPUBackend>(output_idx),
                   stream, use_copy_kernel);
  }
  if (sync) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

void daliOutputCopySamples(daliPipelineHandle *pipe_handle, void **dsts, int output_idx,
                           device_type_t dst_type, cudaStream_t stream, unsigned int flags) {
  dali::DomainTimeRange tr("[DALI][C API] daliOutputCopySamples", dali::DomainTimeRange::kGreen);

  bool is_pinned = flags & DALI_ext_pinned;
  bool sync = flags & DALI_ext_force_sync;
  bool use_copy_kernel = flags & DALI_use_copy_kernel;
  auto dst_mem_kind = GetMemKind(dst_type, is_pinned);

  dali::DeviceWorkspace *ws = reinterpret_cast<dali::DeviceWorkspace *>(pipe_handle->ws);
  assert(ws != nullptr);

  auto &type_info = dali::TypeTable::GetTypeInfo(dali::DALIDataType::DALI_UINT8);
  if (ws->OutputIsType<dali::CPUBackend>(output_idx)) {
    CopyToExternal(dsts, dst_mem_kind, ws->OutputRef<dali::CPUBackend>(output_idx),
                   stream, use_copy_kernel);
  } else {
    CopyToExternal(dsts, dst_mem_kind, ws->OutputRef<dali::GPUBackend>(output_idx),
                   stream, use_copy_kernel);
  }
  if (sync) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}


void daliCopyTensorNTo(daliPipelineHandle *pipe_handle, void *dst, int output_id,
                    device_type_t dst_type, cudaStream_t stream, int non_blocking) {
  DALI_WARN("Warning: daliCopyTensorNTo is now deprecated. Use daliOutputCopy instead.");

  unsigned int flags = DALI_ext_default;
  if (non_blocking == 0)
    flags |= DALI_ext_force_sync;

  daliOutputCopy(pipe_handle, dst, output_id, dst_type, stream, flags);
}

void daliCopyTensorListNTo(daliPipelineHandle *pipe_handle, void *dst, int output_id,
                           device_type_t dst_type, cudaStream_t stream, int non_blocking) {
  DALI_WARN("Warning: daliCopyTensorListNTo is now deprecated. Use daliOutputCopy instead.");

  unsigned int flags = DALI_ext_default;
  if (non_blocking == 0)
    flags |= DALI_ext_force_sync;

  daliOutputCopy(pipe_handle, dst, output_id, dst_type, stream, flags);
}

void daliDeletePipeline(daliPipelineHandle* pipe_handle) {
  dali::Pipeline *pipeline = reinterpret_cast<dali::Pipeline *>(pipe_handle->pipe);
  dali::DeviceWorkspace *ws = reinterpret_cast<dali::DeviceWorkspace *>(pipe_handle->ws);
  auto *bs_map = reinterpret_cast<batch_size_map_t *>(pipe_handle->batch_sizes_map);
  DALI_ENFORCE(pipeline != nullptr && ws != nullptr, "Pipeline already deleted");
  if (pipe_handle->copy_stream) {
    CUDA_CALL(cudaStreamDestroy(pipe_handle->copy_stream));
  }
  pipe_handle->copy_stream = nullptr;
  delete ws;
  delete pipeline;
  delete bs_map;
  pipe_handle->ws = nullptr;
  pipe_handle->pipe = nullptr;
  pipe_handle->batch_sizes_map = nullptr;
}

void daliLoadLibrary(const char* lib_path) {
    dali::PluginManager::LoadLibrary(lib_path);
}

void daliGetReaderMetadata(daliPipelineHandle* pipe_handle, const char *reader_name,
                           daliReaderMetadata* meta) {
  DALI_ENFORCE(meta, "Provided pointer to meta cannot be NULL.");
  dali::Pipeline* pipeline = reinterpret_cast<dali::Pipeline*>(pipe_handle->pipe);
  dali::ReaderMeta returned_meta = pipeline->GetReaderMeta(reader_name);
  meta->epoch_size = returned_meta.epoch_size;
  meta->epoch_size_padded = returned_meta.epoch_size_padded;
  meta->number_of_shards = returned_meta.number_of_shards;
  meta->shard_id = returned_meta.shard_id;
  meta->pad_last_batch = returned_meta.pad_last_batch;
  meta->stick_to_shard = returned_meta.stick_to_shard;
}

dali_backend_t daliGetOperatorBackend(daliPipelineHandle* pipe_handle, const char *operator_name) {
  dali::Pipeline* pipeline = reinterpret_cast<dali::Pipeline*>(pipe_handle->pipe);
  auto *node = pipeline->GetOperatorNode(operator_name);
  switch (node->op_type) {
    case dali::OpType::CPU:
      return dali_backend_t::DALI_BACKEND_CPU;
    case dali::OpType::GPU:
      return dali_backend_t::DALI_BACKEND_GPU;
    case dali::OpType::MIXED:
      return dali_backend_t::DALI_BACKEND_MIXED;
    default:
      DALI_FAIL("Invalid operator type.");
  }
}

void daliGetExecutorMetadata(daliPipelineHandle* pipe_handle, daliExecutorMetadata **operator_meta,
                             size_t *operator_meta_num) {
  dali::Pipeline* pipeline = reinterpret_cast<dali::Pipeline*>(pipe_handle->pipe);
  auto returned_meta = pipeline->GetExecutorMeta();
  *operator_meta_num = returned_meta.size();
  *operator_meta = static_cast<daliExecutorMetadata*>(malloc(sizeof(daliExecutorMetadata) *
                                                     returned_meta.size()));

  int i = 0;
  for (const auto &stat : returned_meta) {
    auto op_name_size = stat.first.size();
    auto &op_meta = (*operator_meta)[i];
    op_meta.operator_name = static_cast<char*>(malloc(sizeof(char) * (op_name_size + 1)));
    stat.first.copy(op_meta.operator_name, op_name_size);
    op_meta.operator_name[op_name_size] = '\0';

    auto num_outputs = stat.second.size();
    op_meta.out_num = num_outputs;
    op_meta.real_size = static_cast<size_t*>(malloc(sizeof(size_t) * num_outputs));
    op_meta.max_real_size = static_cast<size_t*>(malloc(sizeof(size_t) * num_outputs));
    op_meta.reserved = static_cast<size_t*>(malloc(sizeof(size_t) * num_outputs));
    op_meta.max_reserved = static_cast<size_t*>(malloc(sizeof(size_t) * num_outputs));

    for (size_t j = 0; j < num_outputs; ++j) {
      const auto &entry = stat.second[j];
      op_meta.real_size[j] = entry.real_size;
      op_meta.max_real_size[j] = entry.max_real_size;
      op_meta.reserved[j] = entry.reserved;
      op_meta.max_reserved[j] = entry.max_reserved;
    }
    ++i;
  }
}

void daliFreeExecutorMetadata(daliExecutorMetadata *operator_meta, size_t operator_meta_num) {
  for (size_t i = 0; i < operator_meta_num; ++i) {
    free(operator_meta[i].operator_name);
    free(operator_meta[i].real_size);
    free(operator_meta[i].max_real_size);
    free(operator_meta[i].reserved);
    free(operator_meta[i].max_reserved);
  }
  free(operator_meta);
}
