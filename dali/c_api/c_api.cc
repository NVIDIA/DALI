// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "dali/core/common.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/format.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/mm/default_resources.h"
#include "dali/pipeline/init.h"

#include "dali/pipeline/pipeline.h"
#include "dali/plugin/plugin_manager.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/copy_to_external.h"
#include "dali/pipeline/operator/checkpointing/checkpoint.h"

#include "dali/c_api.h"  // NOLINT [build/include]

// Sanity check that the values of the flags are consistent with the ExecutorType enum.
namespace dali {
namespace {
static_assert(static_cast<ExecutorType>(DALI_EXEC_IS_PIPELINED) == ExecutorType::PipelinedFlag);
static_assert(static_cast<ExecutorType>(DALI_EXEC_IS_ASYNC) == ExecutorType::AsyncFlag);
static_assert(static_cast<ExecutorType>(DALI_EXEC_IS_SEPARATED) == ExecutorType::SeparatedFlag);
static_assert(static_cast<ExecutorType>(DALI_EXEC_IS_DYNAMIC) == ExecutorType::DynamicFlag);
}  // namespace
}  // namespace dali

using dali::AccessOrder;
using dali::CPUBackend;
using dali::GPUBackend;

/**
 * Maps operator name to the batch size set prior to daliSetExternal... call.
 *
 * Typically, this operator will be BatchSizeProvider.
 * Negative values denote max batch size (default state).
 * Typical usage:
 * auto *batch_size_map = reinterpret_cast<batch_size_map_t *>(handle->batch_size_map);
 */
using batch_size_map_t = std::unordered_map<std::string /* op_name */, int /* batch_size */>;

/**
 * Maps operator name to the data_id set prior to daliSetExternal... call.
 *
 * Usually the operator with given `op_name` will be an InputOperator.
 * Typical usage:
 * auto *data_id_map = reinterpret_cast<data_id_map_t *>(handle->data_id_map);
 */
using data_id_map_t = std::unordered_map<std::string /* op_name */, std::string /* data_id */>;

/**
 * @brief Aggregates a DALI pipeline and auxiliary objects
 */
struct DALIPipeline {
  std::unique_ptr<dali::Pipeline> pipeline;
  dali::Workspace workspace;
  batch_size_map_t batch_size_map;
  data_id_map_t data_id_map;
  dali::CUDAStreamLease copy_stream;
};

/** Temporary workaround for backward compatibility
 *
 * In the long run, daliPipeline_t is going to be replaced with daliPipeline (aka DALIPipeline *)
 * and is going to be passed by value.
 */
typedef daliPipelineHandle *daliPipelineHandle_t;

namespace {

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
 * @brief Extract InputOperatorSettingMode based on the DALI_ext_force_copy and DALI_ext_force_no_copy
 *
 * @param flags Flags typically specified in daliSetExternalInput* functions.
 */
dali::InputOperatorCopyMode GetExternalSourceCopyMode(unsigned int flags) {
  dali::InputOperatorCopyMode copy_mode = dali::InputOperatorCopyMode::DEFAULT;
  DALI_ENFORCE(!((flags & DALI_ext_force_copy) && (flags & DALI_ext_force_no_copy)),
               "External Source cannot be forced to use DALI_ext_force_copy and "
               "DALI_ext_force_no_copy at the same time.");
  if (flags & DALI_ext_force_copy) {
    copy_mode = dali::InputOperatorCopyMode::FORCE_COPY;
  } else if (flags & DALI_ext_force_no_copy) {
    copy_mode = dali::InputOperatorCopyMode::FORCE_NO_COPY;
  }
  return copy_mode;
}

template <typename Backend>
void SetExternalInput(daliPipelineHandle_t pipe_handle, const char *name, const void *data_ptr,
                      dali_data_type_t data_type, const int64_t *shapes, int sample_dim,
                      const char *layout_str, cudaStream_t stream = 0, unsigned int flags = 0) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  auto *bs_map = &(*pipe_handle)->batch_size_map;
  auto *data_id_map = &(*pipe_handle)->data_id_map;
  auto curr_batch_size = PopCurrBatchSize(bs_map, pipeline->max_batch_size(), name);
  std::vector<int64_t> shapes_tmp(shapes, shapes + sample_dim * curr_batch_size);
  dali::TensorListShape<> tl_shape(std::move(shapes_tmp), curr_batch_size, sample_dim);
  dali::TensorLayout layout{};
  if (layout_str != nullptr) {
    layout = dali::TensorLayout(layout_str);
  }
  dali::TensorList<Backend> data;
  auto type_id = static_cast<dali::DALIDataType>(data_type);
  auto elem_sizeof = dali::TypeTable::GetTypeInfo(type_id).size();
  // We cast away the const from data_ptr, as there is no other way of passing it to the
  // TensorList, as we must also set the shape and type metadata.
  // It is passed further as const TensorList, so it's data cannot be modified.
  AccessOrder order;
  if (std::is_same_v<Backend, GPUBackend> || (flags & DALI_ext_pinned))
    order = AccessOrder(stream);
  else
    order = AccessOrder::host();
  // We do not support feeding memory cross-device, it is assumed it's on the current device
  // that is tied to the pipeline.
  int device_id = pipeline->device_id();
  data.ShareData(std::shared_ptr<void>(const_cast<void *>(data_ptr), [](void *) {}),
                 tl_shape.num_elements() * elem_sizeof, flags & DALI_ext_pinned, tl_shape, type_id,
                 device_id, order);
  data.SetLayout(layout);

  auto data_id = data_id_map->extract(name);

  pipeline->SetExternalInput(name, data, order, flags & DALI_ext_force_sync,
                             flags & DALI_use_copy_kernel, GetExternalSourceCopyMode(flags),
                             data_id ? std::make_optional(data_id.mapped()) : std::nullopt);
}


template<typename Backend>
void SetExternalInputTensors(daliPipelineHandle_t pipe_handle, const char *name,
                             const void *const *data_ptr, dali_data_type_t data_type,
                             const int64_t *shapes, int64_t sample_dim, const char *layout_str,
                             cudaStream_t stream = 0, unsigned int flags = 0) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  auto *bs_map = &(*pipe_handle)->batch_size_map;
  auto *data_id_map = &(*pipe_handle)->data_id_map;
  auto curr_batch_size = PopCurrBatchSize(bs_map, pipeline->max_batch_size(), name);
  std::vector<int64_t> shapes_tmp(shapes, shapes + sample_dim * curr_batch_size);
  dali::TensorListShape<> tl_shape(std::move(shapes_tmp), curr_batch_size, sample_dim);
  dali::TensorLayout layout{};
  if (layout_str != nullptr) {
    layout = dali::TensorLayout(layout_str);
  }
  auto type_id = static_cast<dali::DALIDataType>(data_type);
  auto elem_sizeof = dali::TypeTable::GetTypeInfo(type_id).size();

  AccessOrder order;
  if (std::is_same_v<Backend, GPUBackend> || (flags & DALI_ext_pinned))
    order = AccessOrder(stream);
  else
    order = AccessOrder::host();

  // We do not support feeding memory cross-device, it is assumed it's on the current device
  // that is tied to the pipeline.
  int device_id = pipeline->device_id();

  dali::TensorList<Backend> data(curr_batch_size);
  data.set_pinned(flags & DALI_ext_pinned);
  data.set_sample_dim(sample_dim);
  data.set_type(type_id);
  data.set_device_id(device_id);
  data.set_order(order);
  data.SetLayout(layout);

  for (int i = 0; i < curr_batch_size; i++) {
    // We cast away the const from data_ptr, as there is no other way of passing it to the
    // Tensor as we must also set the shape and type metadata.
    // The vector that we pass to pipeline is const.
    std::shared_ptr<void> ptr(const_cast<void *>(data_ptr[i]), [](void *){});  // no deleter
    data.SetSample(i, ptr, tl_shape[i].num_elements() * elem_sizeof, flags & DALI_ext_pinned,
                   tl_shape[i], type_id, device_id, order, layout);
  }

  auto data_id = data_id_map->extract(name);

  pipeline->SetExternalInput(name, data, order, flags & DALI_ext_force_sync,
                             flags & DALI_use_copy_kernel, GetExternalSourceCopyMode(flags),
                             data_id ? std::make_optional(data_id.mapped()) : std::nullopt);
}

inline dali::mm::memory_kind_id GetMemKind(device_type_t device_type, bool is_pinned) {
  return device_type == device_type_t::GPU
        ? dali::mm::memory_kind_id::device
        : (is_pinned ? dali::mm::memory_kind_id::pinned : dali::mm::memory_kind_id::host);
}

inline std::unique_ptr<DALIPipeline> WrapPipeline(std::unique_ptr<dali::Pipeline> pipeline) {
  auto pipe_wrap = std::make_unique<DALIPipeline>();

  if (pipeline->device_id() >= 0) {
    pipe_wrap->copy_stream = dali::CUDAStreamPool::instance().Get(pipeline->device_id());
  }

  pipe_wrap->pipeline = std::move(pipeline);
  return pipe_wrap;
}

}  // namespace


void daliInitialize() {
  static int init = []() {
    dali::DALIInit(dali::OpSpec("CPUAllocator"),
                   dali::OpSpec("PinnedCPUAllocator"),
                   dali::OpSpec("GPUAllocator"));
    return 0;
  }();
  (void)init;
}


void daliCreatePipeline(daliPipelineHandle *pipe_handle, const char *serialized_pipeline,
                        int length, int max_batch_size, int num_threads, int device_id,
                        int separated_execution, int prefetch_queue_depth,
                        int cpu_prefetch_queue_depth, int gpu_prefetch_queue_depth,
                        int enable_memory_stats) {
  daliCreatePipeline2(pipe_handle, serialized_pipeline, length, max_batch_size, num_threads,
                      device_id, 1, 1, separated_execution, prefetch_queue_depth,
                      cpu_prefetch_queue_depth, gpu_prefetch_queue_depth, enable_memory_stats);
}

DLL_PUBLIC void
daliCreatePipeline2(daliPipelineHandle *pipe_handle, const char *serialized_pipeline, int length,
                    int max_batch_size, int num_threads, int device_id, int pipelined_execution,
                    int async_execution, int separated_execution, int prefetch_queue_depth,
                    int cpu_prefetch_queue_depth, int gpu_prefetch_queue_depth,
                    int enable_memory_stats) {
  dali_exec_flags_t flags = {};
  if (async_execution)  // there's no non-pipelined async executor
    flags = flags | DALI_EXEC_IS_ASYNC | DALI_EXEC_IS_PIPELINED;
  if (pipelined_execution)
    flags = flags | DALI_EXEC_IS_PIPELINED;
  if (separated_execution)
    flags = flags | DALI_EXEC_IS_SEPARATED;
  daliCreatePipeline3(pipe_handle, serialized_pipeline, length,
                      max_batch_size, num_threads, device_id, flags,
                      prefetch_queue_depth, cpu_prefetch_queue_depth, gpu_prefetch_queue_depth,
                      enable_memory_stats);
}

DLL_PUBLIC void
daliCreatePipeline3(daliPipelineHandle *pipe_handle, const char *serialized_pipeline, int length,
                    int max_batch_size, int num_threads, int device_id,
                    dali_exec_flags_t exec_flags, int prefetch_queue_depth,
                    int cpu_prefetch_queue_depth, int gpu_prefetch_queue_depth,
                    int enable_memory_stats) {
  dali::PipelineParams params = dali::MakePipelineParams(max_batch_size, num_threads, device_id);
  params.executor_type = static_cast<dali::ExecutorType>(exec_flags);
  if (exec_flags & DALI_EXEC_IS_SEPARATED) {
    dali::QueueSizes queue_sizes{cpu_prefetch_queue_depth, gpu_prefetch_queue_depth};
    params.prefetch_queue_depths = queue_sizes;
  } else {
    params.prefetch_queue_depths = dali::QueueSizes{prefetch_queue_depth, prefetch_queue_depth};
  }
  params.executor_type = static_cast<dali::ExecutorType>(exec_flags);
  params.enable_memory_stats = enable_memory_stats;

  auto pipeline = std::make_unique<dali::Pipeline>(
      std::string(serialized_pipeline, length), params);
  pipeline->Build();

  *pipe_handle = WrapPipeline(std::move(pipeline)).release();
}

void daliDeserializeDefault(daliPipelineHandle *pipe_handle, const char *serialized_pipeline,
                            int length) {
  auto pipeline = std::make_unique<dali::Pipeline>(std::string(serialized_pipeline, length));
  pipeline->Build();
  *pipe_handle = WrapPipeline(std::move(pipeline)).release();
}


int daliIsDeserializable(const char* serialized_pipeline, int length) {
  auto len = static_cast<size_t>(length);
  return dali::Pipeline::IsDeserializable({serialized_pipeline, len}) ? 0 : 1;
}


int daliGetMaxBatchSize(daliPipelineHandle_t pipe_handle) {
  return (*pipe_handle)->pipeline->max_batch_size();
}

int daliInputFeedCount(daliPipelineHandle_t pipe_handle, const char *input_name) {
  auto &pipeline = (*pipe_handle)->pipeline;
  return pipeline->InputFeedCount(input_name);
}

void daliPrefetch(daliPipelineHandle_t pipe_handle) {
  auto &pipeline = (*pipe_handle)->pipeline;
  pipeline->Prefetch();
}

void daliPrefetchUniform(daliPipelineHandle_t pipe_handle, int queue_depth) {
  auto &pipeline = (*pipe_handle)->pipeline;
  auto sz = pipeline->GetQueueSizes();
  if (queue_depth != sz.cpu_size || queue_depth != sz.gpu_size) {
    DALI_WARN("daliPrefetchUniform is deprecated and setting queue_length different than"
    " the one set in the pipeline has no effect. Use daliPrefetch instead.");
  }
  pipeline->Prefetch();
}


void daliPrefetchSeparate(daliPipelineHandle_t pipe_handle,
                          int cpu_queue_depth, int gpu_queue_depth) {
  auto &pipeline = (*pipe_handle)->pipeline;
  auto sz = pipeline->GetQueueSizes();
  if (cpu_queue_depth != sz.cpu_size || gpu_queue_depth != sz.gpu_size) {
    DALI_WARN("daliPrefetchSeparate is deprecated and setting queue_length different than"
    " the one set in the pipeline has no effect. Use daliPrefetch instead.");
  }
  pipeline->Prefetch();
}


void daliSetExternalInputBatchSize(daliPipelineHandle_t pipe_handle, const char *name,
                                   int batch_size) {
  (*pipe_handle)->batch_size_map[name] = batch_size;
}


void daliSetExternalInputDataId(daliPipelineHandle_t pipe_handle, const char *operator_name,
                                const char *data_id) {
  (*pipe_handle)->data_id_map[operator_name] = data_id;
}


void daliSetExternalInput(daliPipelineHandle_t pipe_handle, const char *name, device_type_t device,
                          const void *data_ptr, dali_data_type_t data_type, const int64_t *shapes,
                          int sample_dim, const char *layout_str, unsigned int flags) {
  daliSetExternalInputAsync(pipe_handle, name, device, data_ptr, data_type, shapes, sample_dim,
                            layout_str, (*pipe_handle)->copy_stream, flags | DALI_ext_force_sync);
}

void daliSetExternalInputAsync(daliPipelineHandle_t pipe_handle, const char *name,
                               device_type_t device, const void *data_ptr,
                               dali_data_type_t data_type, const int64_t *shapes,
                               int sample_dim, const char *layout_str, cudaStream_t stream,
                               unsigned int flags) {
  switch (device) {
    case device_type_t::CPU:
      SetExternalInput<CPUBackend>(pipe_handle, name, data_ptr, data_type, shapes, sample_dim,
                                   layout_str, stream, flags);
      return;
    case device_type_t::GPU:
      SetExternalInput<GPUBackend>(pipe_handle, name, data_ptr, data_type, shapes, sample_dim,
                                   layout_str, stream, flags);
      return;
    default:
      DALI_FAIL(dali::make_string("Unknown device: ", device));
  }
}


void daliSetExternalInputTensors(daliPipelineHandle_t pipe_handle, const char *name,
                                 device_type_t device, const void *const *data_ptr,
                                 dali_data_type_t data_type, const int64_t *shapes,
                                 int64_t sample_dim, const char *layout_str, unsigned int flags) {
  daliSetExternalInputTensorsAsync(pipe_handle, name, device, data_ptr, data_type, shapes,
                                        sample_dim, layout_str, (*pipe_handle)->copy_stream,
                                        flags | DALI_ext_force_sync);
}


void daliSetExternalInputTensorsAsync(daliPipelineHandle_t pipe_handle, const char *name,
                                      device_type_t device, const void *const *data_ptr,
                                      dali_data_type_t data_type, const int64_t *shapes,
                                      int64_t sample_dim, const char *layout_str,
                                      cudaStream_t stream, unsigned int flags) {
  switch (device) {
    case device_type_t::CPU:
      SetExternalInputTensors<CPUBackend>(pipe_handle, name, data_ptr, data_type, shapes,
                                          sample_dim, layout_str, stream, flags);
      return;
    case device_type_t::GPU:
      SetExternalInputTensors<GPUBackend>(pipe_handle, name, data_ptr, data_type, shapes,
                                          sample_dim, layout_str, stream, flags);
      return;
    default:
      DALI_FAIL(dali::make_string("Unknown device: ", device));
  }
}


int daliGetNumExternalInput(daliPipelineHandle_t pipe_handle) {
  return (*pipe_handle)->pipeline->num_inputs();
}


const char *daliGetExternalInputName(daliPipelineHandle_t pipe_handle, int n) {
  return (*pipe_handle)->pipeline->input_name(n).c_str();
}


const char *daliGetExternalInputLayout(daliPipelineHandle_t pipe_handle, const char *name) {
  return (*pipe_handle)->pipeline->GetInputLayout(name).c_str();
}


int daliGetExternalInputNdim(daliPipelineHandle_t pipe_handle, const char *name) {
  return (*pipe_handle)->pipeline->GetInputNdim(name);
}

dali_data_type_t daliGetExternalInputType(daliPipelineHandle_t pipe_handle, const char *name) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  auto type_id = pipeline->GetInputDtype(name);
  return static_cast<dali_data_type_t>(static_cast<int>(type_id));
}

void daliRun(daliPipelineHandle_t pipe_handle) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  pipeline->Run();
}


void daliOutput(daliPipelineHandle_t pipe_handle) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  pipeline->Outputs(&(*pipe_handle)->workspace);
}


void daliShareOutput(daliPipelineHandle_t pipe_handle) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  pipeline->ShareOutputs(&(*pipe_handle)->workspace);
}


void daliOutputRelease(daliPipelineHandle_t pipe_handle) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  pipeline->ReleaseOutputs();
}

int64_t daliOutputHasUniformShape(daliPipelineHandle_t pipe_handle, int i) {
  dali::Workspace* ws = &(*pipe_handle)->workspace;
  if (ws->OutputIsType<CPUBackend>(i)) {
    return is_uniform(ws->Output<CPUBackend>(i).shape());
  } else {
    return is_uniform(ws->Output<GPUBackend>(i).shape());
  }
}

template<typename T>
static int64_t *daliShapeAtHelper(dali::Workspace *ws, int n, int k) {
  int64_t *c_shape = nullptr;
  std::vector<dali::Index> shape;
  const auto &out_tensor_list = ws->Output<T>(n);
  if (k >= 0) {
    auto shape_span = out_tensor_list.tensor_shape_span(k);
    shape = std::vector<dali::Index>(shape_span.begin(), shape_span.end());
  } else {
    auto shape_span = out_tensor_list.tensor_shape_span(0);
    shape = std::vector<dali::Index>(shape_span.begin(), shape_span.end());
    shape.insert(shape.begin(), out_tensor_list.num_samples());
  }

  c_shape = static_cast<int64_t*>(malloc(sizeof(int64_t) * (shape.size() + 1)));
  if (!c_shape) {
    return nullptr;
  }
  c_shape[shape.size()] = 0;
  memcpy(c_shape, &shape[0], shape.size() * sizeof(int64_t));
  return c_shape;
}

static int64_t* daliShapeAtTypedHelper(daliPipelineHandle_t pipe_handle, int n, int k) {
  dali::Workspace* ws = &(*pipe_handle)->workspace;
  if (ws->OutputIsType<CPUBackend>(n)) {
    return daliShapeAtHelper<CPUBackend>(ws, n, k);
  } else {
    return daliShapeAtHelper<GPUBackend>(ws, n, k);
  }
}

int64_t* daliShapeAtSample(daliPipelineHandle_t pipe_handle, int n, int k) {
  return daliShapeAtTypedHelper(pipe_handle, n, k);
}

int64_t* daliShapeAt(daliPipelineHandle_t pipe_handle, int n) {
  return daliShapeAtTypedHelper(pipe_handle, n, -1);
}

template <typename T>
static dali_data_type_t daliTypeAtHelper(dali::Workspace* ws, int n) {
  const auto &out_tensor_list = ws->Output<T>(n);
  auto type_id = out_tensor_list.type();
  return static_cast<dali_data_type_t>(static_cast<int>(type_id));
}

dali_data_type_t daliTypeAt(daliPipelineHandle_t pipe_handle, int n) {
  dali::Workspace* ws = &(*pipe_handle)->workspace;
  if (ws->OutputIsType<CPUBackend>(n)) {
    return daliTypeAtHelper<CPUBackend>(ws, n);
  } else {
    return daliTypeAtHelper<GPUBackend>(ws, n);
  }
}


template <typename T>
static size_t daliNumTensorsHelper(dali::Workspace* ws, int n) {
  return ws->Output<T>(n).num_samples();
}

size_t daliNumTensors(daliPipelineHandle_t pipe_handle, int n) {
  dali::Workspace* ws = &(*pipe_handle)->workspace;
  if (ws->OutputIsType<CPUBackend>(n)) {
    return daliNumTensorsHelper<CPUBackend>(ws, n);
  } else {
    return daliNumTensorsHelper<GPUBackend>(ws, n);
  }
}

template <typename T>
static size_t daliNumElementsHelper(dali::Workspace* ws, int n) {
  return ws->Output<T>(n)._num_elements();
}

size_t daliNumElements(daliPipelineHandle_t pipe_handle, int n) {
  dali::Workspace* ws = &(*pipe_handle)->workspace;
  if (ws->OutputIsType<CPUBackend>(n)) {
    return daliNumElementsHelper<CPUBackend>(ws, n);
  } else {
    return daliNumElementsHelper<GPUBackend>(ws, n);
  }
}

template <typename T>
static size_t daliTensorSizeHelper(dali::Workspace* ws, int n) {
  return ws->Output<T>(n).nbytes();
}

size_t daliTensorSize(daliPipelineHandle_t pipe_handle, int n) {
  dali::Workspace* ws = &(*pipe_handle)->workspace;
  if (ws->OutputIsType<CPUBackend>(n)) {
    return daliTensorSizeHelper<CPUBackend>(ws, n);
  } else {
    return daliTensorSizeHelper<GPUBackend>(ws, n);
  }
}

size_t daliMaxDimTensors(daliPipelineHandle_t pipe_handle, int n) {
  dali::Workspace* ws = &(*pipe_handle)->workspace;
  return ws->GetOutputDim(n);
}

size_t daliGetDeclaredOutputNdim(daliPipelineHandle_t pipe_handle, int n) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  return pipeline->output_ndim(n);
}

dali_data_type_t daliGetDeclaredOutputDtype(daliPipelineHandle_t pipe_handle, int n) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  return static_cast<dali_data_type_t>(static_cast<int>(pipeline->output_dtype(n)));
}

unsigned daliGetNumOutput(daliPipelineHandle_t pipe_handle) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  return pipeline->num_outputs();
}

const char *daliGetOutputName(daliPipelineHandle_t pipe_handle, int id) {
  auto *pipeline = (*pipe_handle)->pipeline.get();
  return pipeline->output_name(id).c_str();
}


device_type_t daliGetOutputDevice(daliPipelineHandle_t pipe_handle, int id) {
  dali::Pipeline *pipeline = (*pipe_handle)->pipeline.get();
  return static_cast<device_type_t>(pipeline->output_device(id));
}


int daliHasOperatorTrace(daliPipelineHandle_t pipe_handle, const char *operator_name,
                         const char *trace_name) {
  auto *ws = &(*pipe_handle)->workspace;
  try {
    auto& traces = ws->GetOperatorTraces(operator_name);
    return traces.find(trace_name) != traces.end() ? 1 : 0;
  } catch (std::out_of_range&) {
    return -1;
  }
}


const char *
daliGetOperatorTrace(daliPipelineHandle_t pipe_handle, const char *operator_name,
                     const char *trace_name) {
  auto *ws = &(*pipe_handle)->workspace;
  if (daliHasOperatorTrace(pipe_handle, operator_name, trace_name)) {
    return ws->GetOperatorTraces(operator_name).at(trace_name).c_str();
  }
  return nullptr;
}


void daliOutputCopy(daliPipelineHandle_t pipe_handle, void *dst, int output_idx,
                    device_type_t dst_type, cudaStream_t stream, unsigned int flags) {
  dali::DomainTimeRange tr("[DALI][C API] daliOutputCopy", dali::DomainTimeRange::kGreen);

  bool is_pinned = flags & DALI_ext_pinned;
  bool host_sync = flags & DALI_ext_force_sync;
  bool use_copy_kernel = flags & DALI_use_copy_kernel;
  auto dst_mem_kind = GetMemKind(dst_type, is_pinned);

  dali::Workspace *ws = &(*pipe_handle)->workspace;
  assert(ws != nullptr);

  AccessOrder wait_order = AccessOrder::host();
  AccessOrder copy_order = AccessOrder::host();

  std::optional<int> dst_dev_id;
  if (dst_type == GPU) {
    if (stream != 0 && stream != cudaStreamLegacy && stream != cudaStreamPerThread) {
      dst_dev_id = dali::DeviceFromStream(stream);
    }
  }

  if (ws->OutputIsType<CPUBackend>(output_idx)) {
    copy_order = dst_type == GPU ? AccessOrder(stream) : AccessOrder::host();
    auto &src = ws->Output<CPUBackend>(output_idx);
    CopyToExternal(dst, dst_mem_kind, dst_dev_id, src, copy_order, use_copy_kernel);
    if (!host_sync)
      wait_order = src.order();  // if the copy order is host, then wait will be no-op
  } else {
    auto &src = ws->Output<GPUBackend>(output_idx);
    copy_order = stream;
    CopyToExternal(dst, dst_mem_kind, dst_dev_id, src, copy_order, use_copy_kernel);
    if (!host_sync)
      wait_order = src.order();
  }
  wait_order.wait(copy_order);
}

void daliOutputCopySamples(daliPipelineHandle_t pipe_handle, void **dsts, int output_idx,
                           device_type_t dst_type, cudaStream_t stream, unsigned int flags) {
  dali::DomainTimeRange tr("[DALI][C API] daliOutputCopySamples", dali::DomainTimeRange::kGreen);

  bool is_pinned = flags & DALI_ext_pinned;
  bool host_sync = flags & DALI_ext_force_sync;
  bool use_copy_kernel = flags & DALI_use_copy_kernel;
  auto dst_mem_kind = GetMemKind(dst_type, is_pinned);

  dali::Workspace *ws = &(*pipe_handle)->workspace;
  assert(ws != nullptr);

  AccessOrder wait_order = AccessOrder::host();
  AccessOrder copy_order = AccessOrder::host();

  std::optional<int> dst_dev_id;
  if (dst_type == GPU) {
    if (stream != 0 && stream != cudaStreamLegacy && stream != cudaStreamPerThread) {
      dst_dev_id = dali::DeviceFromStream(stream);
    }
  }

  if (ws->OutputIsType<CPUBackend>(output_idx)) {
    copy_order = dst_type == GPU ? AccessOrder(stream) : AccessOrder::host();
    auto & src = ws->Output<CPUBackend>(output_idx);
    CopyToExternal(dsts, dst_mem_kind, dst_dev_id, src, copy_order, use_copy_kernel);
    if (!host_sync)
      wait_order = src.order();  // if the copy order is host, then wait will be no-op
  } else {
    auto &src = ws->Output<GPUBackend>(output_idx);
    copy_order = stream;
    CopyToExternal(dsts, dst_mem_kind, dst_dev_id, src, copy_order, use_copy_kernel);
    if (!host_sync)
      wait_order = src.order();
  }
  wait_order.wait(copy_order);
}


void daliCopyTensorNTo(daliPipelineHandle_t pipe_handle, void *dst, int output_id,
                    device_type_t dst_type, cudaStream_t stream, int non_blocking) {
  DALI_WARN("Warning: daliCopyTensorNTo is now deprecated. Use daliOutputCopy instead.");

  unsigned int flags = DALI_ext_default;
  if (non_blocking == 0)
    flags |= DALI_ext_force_sync;

  daliOutputCopy(pipe_handle, dst, output_id, dst_type, stream, flags);
}

void daliCopyTensorListNTo(daliPipelineHandle_t pipe_handle, void *dst, int output_id,
                           device_type_t dst_type, cudaStream_t stream, int non_blocking) {
  DALI_WARN("Warning: daliCopyTensorListNTo is now deprecated. Use daliOutputCopy instead.");

  unsigned int flags = DALI_ext_default;
  if (non_blocking == 0)
    flags |= DALI_ext_force_sync;

  daliOutputCopy(pipe_handle, dst, output_id, dst_type, stream, flags);
}

void daliDeletePipeline(daliPipelineHandle_t pipe_handle) {
  if (!pipe_handle)
    return;

  auto wrap = std::unique_ptr<DALIPipeline>(*pipe_handle);
  if (wrap->copy_stream)
    CUDA_CALL(cudaStreamSynchronize(wrap->copy_stream));
}

void daliLoadLibrary(const char* lib_path) {
  dali::PluginManager::LoadLibrary(lib_path);
}

void daliLoadPluginDirectory(const char* plugin_dir) {
  dali::PluginManager::LoadDirectory(plugin_dir);
}

void daliLoadDefaultPlugins() {
  dali::PluginManager::LoadDefaultPlugins();
}

void daliGetReaderMetadata(daliPipelineHandle_t pipe_handle, const char *reader_name,
                           daliReaderMetadata* meta) {
  DALI_ENFORCE(meta, "Provided pointer to meta cannot be NULL.");
  dali::Pipeline* pipeline = (*pipe_handle)->pipeline.get();
  dali::ReaderMeta returned_meta = pipeline->GetReaderMeta(reader_name);
  meta->epoch_size = returned_meta.epoch_size;
  meta->epoch_size_padded = returned_meta.epoch_size_padded;
  meta->number_of_shards = returned_meta.number_of_shards;
  meta->shard_id = returned_meta.shard_id;
  meta->pad_last_batch = returned_meta.pad_last_batch;
  meta->stick_to_shard = returned_meta.stick_to_shard;
}

dali_backend_t daliGetOperatorBackend(daliPipelineHandle_t pipe_handle, const char *operator_name) {
  dali::Pipeline* pipeline = (*pipe_handle)->pipeline.get();
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

void daliGetExecutorMetadata(daliPipelineHandle_t pipe_handle, daliExecutorMetadata **operator_meta,
                             size_t *operator_meta_num) {
  dali::Pipeline* pipeline = (*pipe_handle)->pipeline.get();
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

void daliReleaseUnusedMemory() {
  dali::mm::ReleaseUnusedMemory();
}

int daliPreallocateDeviceMemory(size_t bytes, int device_id) {
  try {
    dali::mm::PreallocateDeviceMemory(bytes, device_id);
    return 0;
  } catch (const std::bad_alloc &) {
    return -1;
  }
}

int daliPreallocatePinnedMemory(size_t bytes) {
  try {
    dali::mm::PreallocatePinnedMemory(bytes);
    return 0;
  } catch (const std::bad_alloc &) {
    return -1;
  }
}

void *daliAlloc(size_t n) {
  return malloc(n);
}

void daliFree(void *ptr) {
  free(ptr);
}

void daliGetSerializedCheckpoint(
    daliPipelineHandle_t pipe_handle,
    const daliExternalContextCheckpoint *external_context,
    char **checkpoint, size_t *n) {
  DALI_ENFORCE(external_context, "Provided pointer to external context cannot be NULL.");
  auto &pipeline = (*pipe_handle)->pipeline;
  dali::ExternalContextCheckpoint ctx{};
  if (external_context->pipeline_data.data) {
    ctx.pipeline_data = {
      external_context->pipeline_data.data,
      external_context->pipeline_data.size
    };
  }
  if (external_context->iterator_data.data) {
    ctx.iterator_data = {
      external_context->iterator_data.data,
      external_context->iterator_data.size
    };
  }
  std::string cpt = pipeline->GetSerializedCheckpoint(ctx);
  *n = cpt.size();
  *checkpoint = reinterpret_cast<char *>(daliAlloc(cpt.size()));
  DALI_ENFORCE(*checkpoint, "Failed to allocate memory");
  memcpy(*checkpoint, cpt.c_str(), *n);
}

daliExternalContextField daliExternalContextFieldFromString(const std::string &string) {
  daliExternalContextField field;
  auto n = string.size();
  field.data = static_cast<char *>(daliAlloc(n));
  memcpy(field.data, string.c_str(), n);
  field.size = n;
  return field;
}

void daliRestoreFromSerializedCheckpoint(
    daliPipelineHandle *pipe_handle,
    const char *checkpoint, size_t n,
    daliExternalContextCheckpoint *external_context) {
  DALI_ENFORCE(external_context != nullptr,
               "Null external context provided.");
  auto &pipeline = (*pipe_handle)->pipeline;
  auto ctx = pipeline->RestoreFromSerializedCheckpoint({checkpoint, n});
  if (external_context) {
    *external_context = {};
    if (!ctx.pipeline_data.empty()) {
      external_context->pipeline_data = daliExternalContextFieldFromString(ctx.pipeline_data);
    }
    if (!ctx.iterator_data.empty()) {
      external_context->iterator_data = daliExternalContextFieldFromString(ctx.iterator_data);
    }
  }
}

void daliDestroyExternalContextCheckpoint(daliExternalContextCheckpoint *external_context) {
  if (external_context->pipeline_data.data) daliFree(external_context->pipeline_data.data);
  if (external_context->iterator_data.data) daliFree(external_context->iterator_data.data);
}

