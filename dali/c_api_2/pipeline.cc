// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <utility>
#include "dali/c_api_2/pipeline.h"
#include "dali/c_api_2/pipeline_outputs.h"
#include "dali/c_api_2/checkpoint.h"
#include "dali/c_api_2/error_handling.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/pipeline_output_desc.h"
#include "dali/c_api_2/utils.h"
#include "dali/c_api_2/validation.h"

namespace dali::c_api {

PipelineWrapper *ToPointer(daliPipeline_h handle) {
  if (!handle)
    throw NullHandle("Pipeline");
  return static_cast<PipelineWrapper *>(handle);
}

CheckpointWrapper *ToPointer(daliCheckpoint_h handle) {
  if (!handle)
    throw NullHandle("Checkpoint");
  return static_cast<CheckpointWrapper *>(handle);
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
  return pipeline_->GetInputOperators().size();
}

daliPipelineIODesc_t PipelineWrapper::GetInputDesc(int idx) const & {
  auto &inputs = pipeline_->GetInputOperators();
  size_t n = inputs.size();
  if (idx < 0 || static_cast<size_t>(idx) >= n)
    throw std::out_of_range(make_string(
        "The input index ", idx, " is out of range. The valid range is [0..", n-1, "]."));

  if (input_names_.size() != n) {
    input_names_.clear();
    input_names_.reserve(n);
    for (auto it = inputs.begin(); it != inputs.end(); it++) {
      input_names_.push_back(it->first);
    }
  }
  return GetInputDesc(input_names_[idx]);
}

namespace {

template <typename Backend>
void FillPipelineDesc(daliPipelineIODesc_t &desc, const InputOperator<Backend> &inp) {
  int ndim = inp.in_ndim();
  if (ndim >= 0) {
    desc.ndim_present = true;
    desc.ndim = ndim;
  }
  auto dtype = inp.in_dtype();
  if (dtype != DALI_NO_TYPE) {
    desc.dtype_present = true;
    desc.dtype = dtype;
  }
  auto &layout = inp.in_layout();
  if (layout.size())
    desc.layout = layout.c_str();
  else
    desc.layout = nullptr;
}

}  // namespace

daliPipelineIODesc_t PipelineWrapper::GetInputDesc(std::string_view name) const & {
  auto &inputs = pipeline_->GetInputOperators();
  auto it = inputs.find(name);
  if (it == inputs.end())
    throw invalid_key(make_string("The input with the name \"", name, "\" was not found."));

  daliPipelineIODesc_t desc{};
  desc.name = it->first.c_str();
  desc.device = it->second->op_type == OpType::GPU ? DALI_STORAGE_GPU : DALI_STORAGE_CPU;
  auto *op = pipeline_->GetOperator(name);
  if (auto *inp = dynamic_cast<InputOperator<CPUBackend> *>(op))
    FillPipelineDesc(desc, *inp);
  else if (auto *inp = dynamic_cast<InputOperator<GPUBackend> *>(op))
    FillPipelineDesc(desc, *inp);
  else if (auto *inp = dynamic_cast<InputOperator<MixedBackend> *>(op))
    FillPipelineDesc(desc, *inp);
  else
    throw std::logic_error(make_string(
      "Internal error - the operator \"", name, "\" was found in the input operators map, but "
      "it's not an instance of InputOperator<Backend>."));
  return desc;
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

daliPipelineIODesc_t PipelineWrapper::GetOutputDesc(int idx) const & {
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
  desc.layout         = out.layout.c_str();
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

std::unique_ptr<CheckpointWrapper> PipelineWrapper::GetCheckpoint(
      const daliCheckpointExternalData_t *ext) const {
  auto cpt = std::make_unique<CheckpointWrapper>(pipeline_->GetCheckpoint());
  if (ext) {
    cpt->Unwrap()->external_ctx_cpt_.pipeline_data =
        std::string(ext->pipeline_data.data, ext->pipeline_data.size);
    cpt->Unwrap()->external_ctx_cpt_.iterator_data =
        std::string(ext->iterator_data.data, ext->iterator_data.size);
  }
  return cpt;
}

std::string_view PipelineWrapper::SerializeCheckpoint(CheckpointWrapper &chk) const {
  chk.Serialize(*this);
  return chk.Serialized();
}

void CheckpointWrapper::Serialize(const PipelineWrapper &pipeline) {
  if (!serialized_)
    serialized_ = pipeline.Unwrap()->SerializeCheckpoint(cpt_);
}

std::unique_ptr<CheckpointWrapper>
PipelineWrapper::DeserializeCheckpoint(std::string_view serialized) {
  return std::make_unique<CheckpointWrapper>(pipeline_->DeserializeCheckpoint(serialized));
}

void PipelineWrapper::RestoreFromCheckpoint(CheckpointWrapper &chk) {
  pipeline_->RestoreFromCheckpoint(*chk.Unwrap());
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

daliResult_t daliPipelineGetInputCount(daliPipeline_h pipeline, int *out_input_count) {
  DALI_PROLOG();
  auto p = ToPointer(pipeline);
  CHECK_OUTPUT(out_input_count);
  *out_input_count = p->GetInputCount();
  DALI_EPILOG();
}

DALI_API daliResult_t daliPipelineGetInputDescByIdx(
      daliPipeline_h pipeline,
      daliPipelineIODesc_t *out_input_desc,
      int index) {
  DALI_PROLOG();
  auto p = ToPointer(pipeline);
  CHECK_OUTPUT(out_input_desc);
  *out_input_desc = p->GetInputDesc(index);
  DALI_EPILOG();
}

DALI_API daliResult_t daliPipelineGetInputDesc(
      daliPipeline_h pipeline,
      daliPipelineIODesc_t *out_input_desc,
      const char *name) {
  DALI_PROLOG();
  auto p = ToPointer(pipeline);
  CHECK_OUTPUT(out_input_desc);
  NOT_NULL(name);
  *out_input_desc = p->GetInputDesc(name);
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


daliResult_t daliPipelineGetCheckpoint(
      daliPipeline_h pipeline,
      daliCheckpoint_h *out_checkpoint,
      const daliCheckpointExternalData_t *checkpoint_ext) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  CHECK_OUTPUT(out_checkpoint);
  auto chk = pipe->GetCheckpoint(checkpoint_ext);
  *out_checkpoint = chk.release();  // No throwing beyond this point!
  DALI_EPILOG();
}

daliResult_t daliPipelineRestoreCheckpoint(
      daliPipeline_h pipeline,
      daliCheckpoint_h checkpoint) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  auto chk = ToPointer(checkpoint);
  pipe->RestoreFromCheckpoint(*chk);
  DALI_EPILOG();
}

daliResult_t daliPipelineDeserializeCheckpoint(
      daliPipeline_h pipeline,
      daliCheckpoint_h  *out_checkpoint,
      const char *serialized_checkpoint,
      size_t serialized_checkpoint_size) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  CHECK_OUTPUT(out_checkpoint);
  if (!serialized_checkpoint_size) {
    *out_checkpoint = nullptr;
    return DALI_NO_DATA;
  }
  if (!serialized_checkpoint) {
    throw std::invalid_argument("The parameter `serialized_checkpoint` must not be NULL if "
                                "`serialize_checkpoint_size` is nonzero.");
  }

  auto cpt = pipe->DeserializeCheckpoint(
    std::string_view(serialized_checkpoint, serialized_checkpoint_size));

  *out_checkpoint = cpt.release();  // No throwing beyond this point!
  DALI_EPILOG();
}

daliResult_t daliCheckpointGetExternalData(
      daliCheckpoint_h checkpoint,
      daliCheckpointExternalData_t *out_ext_data) {
  DALI_PROLOG();
  auto cpt = ToPointer(checkpoint);
  CHECK_OUTPUT(out_ext_data);
  *out_ext_data = cpt->ExternalData();
  DALI_EPILOG();
}

daliResult_t daliPipelineSerializeCheckpoint(
      daliPipeline_h pipeline,
      daliCheckpoint_h checkpoint,
      const char **out_data,
      size_t *out_size) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  auto cpt = ToPointer(checkpoint);
  CHECK_OUTPUT(out_data);
  CHECK_OUTPUT(out_size);
  auto serialized = pipe->SerializeCheckpoint(*cpt);
  *out_data = serialized.data();
  *out_size = serialized.size();
  DALI_EPILOG();
}

/** Destroys a checkpoint object */
daliResult_t daliCheckpointDestroy(daliCheckpoint_h checkpoint) {
  DALI_PROLOG();
  delete ToPointer(checkpoint);
  DALI_EPILOG();
}

namespace {

std::string_view BackendToString(daliBackend_t backend) {
  switch (backend) {
    case DALI_BACKEND_CPU:   return "cpu";
    case DALI_BACKEND_GPU:   return "gpu";
    case DALI_BACKEND_MIXED: return "mixed";
    default:
      throw std::invalid_argument(dali::make_string(
        "Invalid backend value: ", static_cast<int>(backend)));
  }
}

void AddArgToSpec(dali::OpSpec &spec, const daliArgDesc_t &arg) {
  assert(arg.name != nullptr);  // checked in daliPipelineAddOperator
  std::string_view name = arg.name;

  auto check_not_null = [&](auto *x, auto &&field) {
    dali::c_api::CheckNotNull(x, [&]() {
      return dali::make_string("`", field, "` of argument \"", name, "\"");
    });
  };

  switch (arg.dtype) {
    // --- scalar types ---
    case DALI_INT8:
      spec.AddArg(name, static_cast<int8_t>(arg.ivalue));
      break;
    case DALI_INT16:
      spec.AddArg(name, static_cast<int16_t>(arg.ivalue));
      break;
    case DALI_INT32:
      spec.AddArg(name, static_cast<int32_t>(arg.ivalue));
      break;
    case DALI_INT64:
      spec.AddArg(name, arg.ivalue);
      break;
    case DALI_UINT8:
      spec.AddArg(name, static_cast<uint8_t>(arg.uvalue));
      break;
    case DALI_UINT16:
      spec.AddArg(name, static_cast<uint16_t>(arg.uvalue));
      break;
    case DALI_UINT32:
      spec.AddArg(name, static_cast<uint32_t>(arg.uvalue));
      break;
    case DALI_UINT64:
      spec.AddArg(name, arg.uvalue);
      break;
    case DALI_FLOAT:
      spec.AddArg(name, arg.fvalue);
      break;
    // NOT SUPPORTED / NOT IMPLEMENTED!
    /*case DALI_FLOAT64:
      spec.AddArg(name, arg.dvalue);
      break;*/
    case DALI_BOOL:
      spec.AddArg(name, static_cast<bool>(arg.ivalue));
      break;
    case DALI_STRING:
      check_not_null(arg.str, "arg.str");
      spec.AddArg(name, std::string(arg.str));
      break;
    // --- vector (list) types ---
    case DALI_INT_VEC: {
      if (arg.size > 0)
        check_not_null(arg.arr, "arg.arr");
      auto *d = static_cast<const int *>(arg.arr);
      spec.AddArg(name, std::vector<int>(d, d + arg.size));
      break;
    }
    case DALI_FLOAT_VEC: {
      if (arg.size > 0)
        check_not_null(arg.arr, "arg.arr");
      auto *d = static_cast<const float *>(arg.arr);
      spec.AddArg(name, std::vector<float>(d, d + arg.size));
      break;
    }
    case DALI_BOOL_VEC: {
      if (arg.size > 0)
        check_not_null(arg.arr, "arg.arr");
      auto *d = static_cast<const bool *>(arg.arr);
      spec.AddArg(name, std::vector<bool>(d, d + arg.size));
      break;
    }
    case DALI_STRING_VEC: {
      if (arg.size > 0)
        check_not_null(arg.arr, "arg.arr");
      auto *d = static_cast<const char * const *>(arg.arr);
      std::vector<std::string> sv;
      sv.reserve(arg.size);
      for (int64_t i = 0; i < arg.size; i++) {
        if (!d[i])  // the outer `if` prevents make_string in case of no error
          check_not_null(d[i], dali::make_string("arg.arr[", i, "]"));
        sv.emplace_back(d[i]);
      }
      spec.AddArg(name, std::move(sv));
      break;
    }
    default:
      throw std::invalid_argument(dali::make_string(
        "Unsupported argument `dtype`: ", static_cast<int>(arg.dtype),
        " in argument \"", arg.name, "\"."));
  }
}

}  // namespace

daliResult_t daliPipelineAddExternalInput(
      daliPipeline_h pipeline,
      const daliPipelineIODesc_t *input_desc) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  NOT_NULL(input_desc);
  NOT_NULL(input_desc->name);
  std::string device_str = input_desc->device == DALI_STORAGE_GPU ? "gpu" : "cpu";
  daliDataType_t dtype = input_desc->dtype_present ? input_desc->dtype : DALI_NO_TYPE;
  int ndim = input_desc->ndim_present ? input_desc->ndim : -1;
  const char *layout = input_desc->layout ? input_desc->layout : "";
  pipe->Unwrap()->AddExternalInput(input_desc->name, device_str, dtype, ndim, layout);
  DALI_EPILOG();
}

daliResult_t daliPipelineAddOperator(
      daliPipeline_h pipeline,
      const daliOperatorDesc_t *op_desc) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  NOT_NULL(op_desc);
  NOT_NULL(op_desc->schema_name);
  if (op_desc->num_inputs > 0)
    NOT_NULL(op_desc->inputs);
  if (op_desc->num_outputs > 0)
    NOT_NULL(op_desc->outputs);
  if (op_desc->num_arg_inputs > 0)
    NOT_NULL(op_desc->arg_inputs);
  if (op_desc->num_args > 0)
    NOT_NULL(op_desc->args);
  dali::OpSpec spec(op_desc->schema_name);
  spec.AddArg("device", std::string(BackendToString(op_desc->backend)));
  for (int i = 0; i < op_desc->num_inputs; i++) {
    dali::c_api::CheckNotNull(op_desc->inputs[i].name, [i]() {
      return dali::make_string("`op_desc->inputs[", i, "].name`");
    });
    spec.AddInput(op_desc->inputs[i].name,
                  static_cast<dali::StorageDevice>(op_desc->inputs[i].device_type));
  }
  for (int i = 0; i < op_desc->num_outputs; i++) {
    dali::c_api::CheckNotNull(op_desc->outputs[i].name, [i]() {
      return dali::make_string("`op_desc->outputs[", i, "].name`");
    });
    spec.AddOutput(op_desc->outputs[i].name,
                   static_cast<dali::StorageDevice>(op_desc->outputs[i].device_type));
  }
  // Argument inputs need to be added after regular inputs
  for (int i = 0; i < op_desc->num_arg_inputs; i++) {
    dali::c_api::CheckNotNull(op_desc->arg_inputs[i].arg_name, [i]() {
      return dali::make_string(
        "`arg_input[", i, "].arg_name`");
      });
    dali::c_api::CheckNotNull(op_desc->arg_inputs[i].input_name, [i]() {
      return dali::make_string(
        "`arg_input[", i, "].input_name`");
      });
    spec.AddArgumentInput(op_desc->arg_inputs[i].arg_name,
                          op_desc->arg_inputs[i].input_name);
  }
  for (int i = 0; i < op_desc->num_args; i++) {
    dali::c_api::CheckNotNull(op_desc->args[i].name, [i]() {
      return dali::make_string("`op_desc->args[", i, "].name`");
    });
    AddArgToSpec(spec, op_desc->args[i]);
  }
  if (op_desc->instance_name && op_desc->instance_name[0] != '\0')
    pipe->Unwrap()->AddOperator(spec, op_desc->instance_name);
  else
    pipe->Unwrap()->AddOperator(spec);
  DALI_EPILOG();
}

daliResult_t daliPipelineSetOutputs(
      daliPipeline_h pipeline,
      int num_outputs,
      const daliPipelineIODesc_t *outputs) {
  DALI_PROLOG();
  auto pipe = ToPointer(pipeline);
  dali::c_api::CheckArg(num_outputs >= 0, "`num_outputs` must not be negative");
  if (num_outputs == 0)
    return DALI_SUCCESS;  // nothing to do

  NOT_NULL(outputs);
  std::vector<dali::PipelineOutputDesc> descs;
  descs.reserve(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    NOT_NULL(outputs[i].name);
    dali::PipelineOutputDesc desc;
    desc.name   = outputs[i].name;
    desc.device = static_cast<dali::StorageDevice>(outputs[i].device);
    if (outputs[i].dtype_present) desc.dtype = outputs[i].dtype;
    if (outputs[i].ndim_present)  desc.ndim  = outputs[i].ndim;
    if (outputs[i].layout)        desc.layout = outputs[i].layout;
    descs.push_back(std::move(desc));
  }
  pipe->Unwrap()->SetOutputDescs(std::move(descs));
  DALI_EPILOG();
}
