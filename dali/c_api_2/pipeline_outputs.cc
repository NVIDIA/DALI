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
#include "dali/c_api_2/validation.h"
#include "dali/pipeline/pipeline.h"

namespace dali::c_api {

PipelineOutputs *ToPointer(daliPipelineOutputs_h handle) {
  if (!handle)
    throw NullHandle("PipelineOutputs");
  return static_cast<PipelineOutputs *>(handle);
}

PipelineOutputs::PipelineOutputs(Pipeline *pipe, AccessOrder order) {
  ws_.set_output_order(order);
  pipe->ShareOutputs(&ws_);
  producer_ = pipe;
  output_wrappers_.resize(ws_.NumOutput());
}

PipelineOutputs::~PipelineOutputs() {
  if (producer_)
    producer_->ReleaseOutputs();
}

span<daliOperatorTrace_t> PipelineOutputs::GetTraces() {
  if (!traces_.has_value()) {
    traces_.emplace();
    if (auto iter_data = ws_.GetIterationData()) {
      auto &trace_map = iter_data->operator_traces.GetRef();
      for (auto &[op, traces] : trace_map) {
        for (auto &[name, value] : traces) {
          traces_->push_back({ op.c_str(), name.c_str(), value.c_str() });
        }
      }
    }
  }
  return make_span(*traces_);
}

std::optional<std::string_view>
PipelineOutputs::GetTrace(std::string_view op_name, std::string_view trace_name) const {
  auto &t = ws_.GetOperatorTraces(op_name);
  auto it = t.find(trace_name);
  if (it != t.end())
    return it->second;
  return std::nullopt;
}

RefCountedPtr<ITensorList> PipelineOutputs::Get(int index) {
  ValidateOutputIdx(index);

  if (!output_wrappers_[index]) {
    if (ws_.OutputIsType<CPUBackend>(index))
      output_wrappers_[index] = Wrap(ws_.OutputPtr<CPUBackend>(index));
    else if (ws_.OutputIsType<GPUBackend>(index))
      output_wrappers_[index] = Wrap(ws_.OutputPtr<GPUBackend>(index));
    else
      assert(!"Impossible output backend encountered.");
  }
  assert(output_wrappers_[index]->RefCount() >= 1);
  return output_wrappers_[index];
}

}  // namespace dali::c_api

using namespace dali::c_api;  // NOLINT

daliResult_t daliPipelineOutputsDestroy(daliPipelineOutputs_h h) {
  DALI_PROLOG();
  delete ToPointer(h);
  DALI_EPILOG();
}

daliResult_t daliPipelineOutputsGet(daliPipelineOutputs_h outputs, daliTensorList_h *out, int idx) {
  DALI_PROLOG();
  auto *outs = ToPointer(outputs);
  CHECK_OUTPUT(out);
  auto ptr = outs->Get(idx);
  *out = ptr.release();  // no throwing beyond this point
  assert(static_cast<ITensorList *>(*out)->RefCount() > 1);
  DALI_EPILOG();
}

daliResult_t daliPipelineOutputsGetTrace(
      daliPipelineOutputs_h outputs,
      const char **out_trace,
      const char *operator_name,
      const char *trace_name) {
  DALI_PROLOG();
  auto *outs = ToPointer(outputs);
  CHECK_OUTPUT(out_trace);
  NOT_NULL(operator_name);
  NOT_NULL(trace_name);
  auto trace = outs->GetTrace(operator_name, trace_name);
  if (!trace) {
    *out_trace = nullptr;
    return DALI_NO_DATA;
  }
  *out_trace = trace->data();
  DALI_EPILOG();
}

daliResult_t daliPipelineOutputsGetTraces(
      daliPipelineOutputs_h outputs,
      const daliOperatorTrace_t **out_traces,
      int *out_trace_count) {
  DALI_PROLOG();
  auto *outs = ToPointer(outputs);
  CHECK_OUTPUT(out_traces);
  CHECK_OUTPUT(out_trace_count);
  auto traces = outs->GetTraces();
  *out_traces = traces.data();
  *out_trace_count = traces.size();
  DALI_EPILOG();
}
