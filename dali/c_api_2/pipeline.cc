// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/c_api_2/pipeline.h"
#include "dali/c_api_2/pipeline_outputs.h"
#include "dali/c_api_2/error_handling.h"
#include "dali/pipeline/pipeline.h"
#include "dali/c_api_2/utils.h"
#include "dali/c_api_2/validation.h"

namespace dali::c_api {

PipelineWrapper *ToPointer(daliPipeline_h handle) {
  if (!handle)
    throw NullHandle("Pipeline");
  return static_cast<PipelineWrapper *>(handle);
}

PipelineParams ToCppParams(const daliPipelineParams_t &params) {
  PipelineParams cpp_params = {};

  if (params.max_batch_size_present)
    cpp_params.max_batch_size = params.max_batch_size;
  if (params.num_threads_present)
    cpp_params.num_threads = params.num_threads;
  if (params.device_id_present)
    cpp_params.device_id = params.device_id;
  if (params.seed_present)
    cpp_params.seed = params.seed;
  if (params.exec_type_present)
    cpp_params.executor_type = static_cast<ExecutorType>(params.exec_type);
  if (params.exec_flags_present)
    cpp_params.executor_flags = static_cast<ExecutorFlags>(params.exec_flags);

  if (params.prefetch_queue_depths_present)
    cpp_params.prefetch_queue_depths = QueueSizes{params.prefetch_queue_depths.cpu,
                                                  params.prefetch_queue_depths.gpu};

  if (params.enable_checkpointing_present)
    cpp_params.enable_checkpointing = params.enable_checkpointing;
  if (params.enable_memory_stats_present)
    cpp_params.enable_memory_stats = params.enable_memory_stats;
  if (params.bytes_per_sample_hint_present)
    cpp_params.bytes_per_sample_hint = params.bytes_per_sample_hint;
  return cpp_params;
}

PipelineWrapper::PipelineWrapper(const daliPipelineParams_t &params) {
  pipeline_ = std::make_unique<Pipeline>(ToCppParams(params));
}

PipelineWrapper::PipelineWrapper(
      const void *serialized,
      size_t length,
      const daliPipelineParams_t &params) {
  pipeline_ = std::make_unique<Pipeline>(
    std::string(static_cast<const char *>(serialized), length),
    ToCppParams(params));
}

PipelineWrapper::~PipelineWrapper() = default;

std::unique_ptr<PipelineOutputs>
PipelineWrapper::PopOutputs(AccessOrder order) {
  return std::make_unique<PipelineOutputs>(pipeline_.get(), order);
}

void PipelineWrapper::Build() {
  pipeline_->Build();
}

void PipelineWrapper::Run() {
  pipeline_->Run();
}

void PipelineWrapper::Prefetch() {
  pipeline_->Prefetch();
}

int PipelineWrapper::GetInputCount() const {
  return pipeline_->GetInputNdim(
}

int PipelineWrapper::GetFeedCount(std::string_view input_name) {
  return pipeline_->InputFeedCount(input_name);
}

void PipelineWrapper::FeedInput(
    std::string_view input_name,
    const ITensorList *input_data,
    std::optional<std::string_view> data_id,
    daliFeedInputFlags_t options,
    AccessOrder order) {
  assert(input_data);
  if (input_data->GetBufferPlacement().device_type == DALI_STORAGE_CPU) {
    FeedInputImpl(
      input_name,
      *input_data->Unwrap<CPUBackend>(),
      data_id,
      options,
      order);
  } else if (input_data->GetBufferPlacement().device_type == DALI_STORAGE_GPU) {
    FeedInputImpl(
      input_name,
      *input_data->Unwrap<GPUBackend>(),
      data_id,
      options,
      order);
  } else {
    assert(!"Impossible device type encountered.");
  }
}

int PipelineWrapper::GetOutputCount() const {
  return pipeline_->output_descs().size();
}

daliPipelineIODesc_t PipelineWrapper::GetOutputDesc(int idx) const {
  auto &outputs = pipeline_->output_descs();
  int nout = outputs.size();
  if (idx < 0 || idx >= nout)
    throw std::out_of_range(make_string(
        "The output index ", idx, " is out of range. "
        "Valid range is [0..", nout-1, "]."));
  auto &out = outputs[idx];
  daliPipelineIODesc_t desc{};
  desc.device         = static_cast<daliStorageDevice_t>(out.device);
  desc.dtype          = out.dtype;
  desc.dtype_present  = out.dtype != DALI_NO_TYPE;
  desc.name           = out.name.c_str();
  desc.ndim           = out.ndim;
  desc.ndim_present   = out.ndim >= 0;
  return desc;
}

template <typename Backend>
void PipelineWrapper::FeedInputImpl(
      std::string_view input_name,
      const TensorList<Backend> &tl,
      std::optional<std::string_view> data_id,
      daliFeedInputFlags_t options,
      AccessOrder order) {
  InputOperatorCopyMode copy_mode = InputOperatorCopyMode::DEFAULT;

  if (options & DALI_FEED_INPUT_FORCE_COPY) {
    if (options & DALI_FEED_INPUT_NO_COPY)
      throw std::invalid_argument("DALI_FEED_INPUT_FORCE_COPY and DALI_FEED_INPUT_NO_COPY"
                                  " must not be used together.");
    copy_mode = InputOperatorCopyMode::FORCE_COPY;
  } else if (options & DALI_FEED_INPUT_NO_COPY) {
    copy_mode = InputOperatorCopyMode::FORCE_NO_COPY;
  }

  pipeline_->SetExternalInput(
    std::string(input_name),  // TODO(michalz): switch setting input to string_view
    tl,
    order ? order : tl.order(),
    options & DALI_FEED_INPUT_SYNC,
    options & DALI_FEED_INPUT_USE_COPY_KERNEL,
    copy_mode,
    data_id ? std::optional<std::string>(std::in_place, *data_id) : std::nullopt);
}

}  // namespace dali::c_api

using namespace dali::c_api;  // NOLINT
using dali::AccessOrder;

daliResult_t daliPipelineCreate(
      daliPipeline_h *out_pipe_handle,
      const daliPipelineParams_t *params) {
  DALI_PROLOG();
  CHECK_OUTPUT(out_pipe_handle);
  NOT_NULL(params);

  *out_pipe_handle = new PipelineWrapper(*params);
  DALI_EPILOG();
}

daliResult_t daliPipelineDestroy(daliPipeline_h pipeline) {
  DALI_PROLOG();
  delete ToPointer(pipeline);
  DALI_EPILOG();
}

daliResult_t daliPipelineDeserialize(
      daliPipeline_h *out_pipe_handle,
      const void *serialized_pipeline,
      size_t serialized_pipeline_size,
      const daliPipelineParams_t *param_overrides) {
  DALI_PROLOG();
  CHECK_OUTPUT(out_pipe_handle);
  NOT_NULL(serialized_pipeline);
  NOT_NULL(param_overrides);

  *out_pipe_handle = new PipelineWrapper(
      serialized_pipeline,
      serialized_pipeline_size,
      *param_overrides);
  DALI_EPILOG();
}

daliResult_t daliPipelineBuild(daliPipeline_h pipeline) {
  DALI_PROLOG();
  ToPointer(pipeline)->Build();
  DALI_EPILOG();
}

daliResult_t daliPipelinePrefetch(daliPipeline_h pipeline) {
  DALI_PROLOG();
  ToPointer(pipeline)->Prefetch();
  DALI_EPILOG();
}

daliResult_t daliPipelineRun(daliPipeline_h pipeline) {
  DALI_PROLOG();
  ToPointer(pipeline)->Run();
  DALI_EPILOG();
}

daliResult_t daliPipelineGetFeedCount(
      daliPipeline_h pipeline,
      int *out_feed_count,
      const char *input_name) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  CHECK_OUTPUT(out_feed_count);
  NOT_NULL(input_name);
  *out_feed_count = pipe->GetFeedCount(input_name);
  DALI_EPILOG();
}

daliResult_t daliPipelineFeedInput(
      daliPipeline_h pipeline,
      const char *input_name,
      daliTensorList_h input_data,
      const char *data_id,
      daliFeedInputFlags_t options,
      const cudaStream_t *stream) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  NOT_NULL(input_name);
  pipe->FeedInput(
    input_name,
    ToPointer(input_data),
    ToOptionalString(data_id),
    options,
    stream ? AccessOrder(*stream) : AccessOrder());
  DALI_EPILOG();
}

daliResult_t daliPipelineGetOutputCount(daliPipeline_h pipeline, int *out_count) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  CHECK_OUTPUT(out_count);
  *out_count = pipe->GetOutputCount();
  DALI_EPILOG();
}

daliResult_t daliPipelineGetOutputDesc(
      daliPipeline_h pipeline,
      daliPipelineIODesc_t *out_desc,
      int index) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  CHECK_OUTPUT(out_desc);
  *out_desc = pipe->GetOutputDesc(index);
  DALI_EPILOG();
}

daliResult_t daliPipelinePopOutputs(daliPipeline_h pipeline, daliPipelineOutputs_h *out) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  CHECK_OUTPUT(out);
  *out = pipe->PopOutputs().release();
  DALI_EPILOG();
}

daliResult_t daliPipelinePopOutputsAsync(
        daliPipeline_h pipeline,
        daliPipelineOutputs_h *out,
        cudaStream_t stream) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  CHECK_OUTPUT(out);
  *out = pipe->PopOutputs(stream).release();
  DALI_EPILOG();
}
