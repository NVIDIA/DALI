// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>

#include "dali/core/common.h"
#include "dali/core/util.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/builtin/conditional/merge.h"
#include "dali/pipeline/operator/builtin/conditional/split_merge.h"

namespace dali {


template <typename Backend>
bool Merge<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  input_sample_count_ = 0;
  int nonzero_input_sample_idx = -1;
  for (int input_group_idx = 0; input_group_idx < kMaxGroups; input_group_idx++) {
    const auto &input = ws.template Input<Backend>(input_group_idx);
    input_sample_count_ += input.num_samples();
    if (nonzero_input_sample_idx < 0 && input.num_samples() > 0) {
      nonzero_input_sample_idx = input_group_idx;
      continue;  // no point in comparing with ourselves
    }
    if (nonzero_input_sample_idx >= 0 && input.num_samples() > 0) {
      const auto &base_input = ws.template Input<Backend>(nonzero_input_sample_idx);
      const char *error_msg_base =
          "Divergent data found in different branches of conditional operation. All paths in "
          "conditional operation are merged into one batch which must have consistent type, "
          "number of dimensions, layout and other metadata. ";

      DALI_ENFORCE(base_input.type() == input.type(),
                   make_string(error_msg_base, "Found distinct types: ", base_input.type(), " and ",
                               input.type(), "."));

      DALI_ENFORCE(
          base_input.shape().sample_dim() == input.shape().sample_dim(),
          make_string(error_msg_base, "Found distinct sample dimensions: ",
                      base_input.shape().sample_dim(), " and ", input.shape().sample_dim(), "."));

      DALI_ENFORCE(base_input.GetLayout() == input.GetLayout(),
                   make_string(error_msg_base, "Found distinct layouts: \"", base_input.GetLayout(),
                               "\" and \"", input.GetLayout(), "\"."));

      // When not pinned, we can have device_id = CPU_ONLY_DEVICE_ID, and for pinned it is the id
      // of an actual device. We handle correct pinnedness in Run.
      if (base_input.is_pinned() == input.is_pinned()) {
        DALI_ENFORCE(base_input.device_id() == input.device_id(),
                     make_string(error_msg_base, "Found distinct device id: ",
                                 base_input.device_id(), " and ", input.device_id(), "."));
      }
    }
  }

  const auto &predicate = ws.ArgumentInput("predicate");
  DALI_ENFORCE(
      input_sample_count_ == predicate.num_samples(),
      make_string("Merge description must cover whole input, got ", input_sample_count_,
                  " input samples and ", predicate.num_samples(), " elements denoting the merge."));
  DALI_ENFORCE(predicate.shape().sample_dim() == 0, "Only scalar indexing is supported.");
  return false;
}


template <typename Backend>
void Merge<Backend>::RunImpl(Workspace &ws) {
  auto &output = ws.template Output<Backend>(0);
  const auto &predicate = ws.ArgumentInput("predicate");
  auto sample_idx_in_input = uniform_array<kMaxGroups>(0);

  WriteTestsDiagnostics(ws);

  if (!pinned_) {
    // We produce pinned data if the executor said so
    pinned_ = output.is_pinned();
    // TODO(klecki): We keep pinedness if it was passed to us - we can consider relying only
    // on static and not run time information and remove the check below.
    for (int input_group = 0; input_group < kMaxGroups; input_group++) {
      pinned_ = *pinned_ || ws.template Input<Backend>(input_group).is_pinned();
    }
  }

  if (*pinned_ || std::is_same_v<Backend, GPUBackend>) {
    CUDA_CALL(cudaGetDevice(&device_id_));
  } else {
    device_id_ = CPU_ONLY_DEVICE_ID;
  }

  // We propagate views only, so just don't care about what is here and reset
  output.Reset();
  for (int input_group = 0; input_group < kMaxGroups; input_group++) {
    const auto &input = ws.template Input<Backend>(input_group);
    if (input.num_samples() > 0) {
      output.set_type(input.type());
      output.set_sample_dim(input.shape().sample_dim());
      output.SetLayout(input.GetLayout());
      break;
    }
  }
  // The pinned (and order) can differ depending on the pipeline graph. Let the executor
  // set the desired one, and we will copy if we don't match. Device_id is different depending on
  // pinnedness of the memory
  output.set_device_id(device_id_);
  output.set_pinned(*pinned_);

  output.SetSize(input_sample_count_);

  for (int output_sample_idx = 0; output_sample_idx < predicate.num_samples();
       output_sample_idx++) {
    int input_group_idx = get_group_index(predicate, output_sample_idx);
    auto &input = ws.template Input<Backend>(input_group_idx);

    // get the index within input group and increment for the next occurrence.
    int input_sample_idx = sample_idx_in_input[input_group_idx];
    sample_idx_in_input[input_group_idx]++;

    if (std::is_same_v<Backend, GPUBackend> || input.is_pinned() == *pinned_) {
      output.SetSample(output_sample_idx, input, input_sample_idx);
    } else {
      // Pessimistic variant, we need to copy.
      // TODO(klecki): Do one allocation, where samples that we share are 0-volumed - this might
      // be perf optimization reducing the number of allocations to 1.
      CopySampleToOutput(output, output_sample_idx, input, input_sample_idx, ws);
    }
  }
  FinalizeCopy(ws);
  WriteTestsDiagnostics(ws);
}

template <>
void Merge<CPUBackend>::CopySampleToOutput(TensorList<CPUBackend> &output, int output_sample_idx,
                                           const TensorList<CPUBackend> &input,
                                           int input_sample_idx, Workspace &ws) {
  auto &tp = ws.GetThreadPool();
  tp.AddWork(
      [&output, &input, output_sample_idx, input_sample_idx](int thread_idx) {
        output.ResizeSample(output_sample_idx, input.shape()[input_sample_idx]);
        output.CopySample(output_sample_idx, input, input_sample_idx, output.order());
      },
      volume(input.tensor_shape_span(input_sample_idx)));
}


template <>
void Merge<CPUBackend>::FinalizeCopy(Workspace &ws) {
  ws.GetThreadPool().RunAll();
}


template <>
void Merge<GPUBackend>::CopySampleToOutput(TensorList<GPUBackend> &output, int output_sample_idx,
                                           const TensorList<GPUBackend> &input,
                                           int input_sample_idx, Workspace &ws) {
  assert(false && "This codepath should not be executed");
}


template <>
void Merge<GPUBackend>::FinalizeCopy(Workspace &ws) {
  // no-op
}


template <typename Backend>
void Merge<Backend>::RegisterTestsDiagnostics() {
  this->RegisterDiagnostic("input_0_pinned", &in_0_pinned_);
  this->RegisterDiagnostic("input_1_pinned", &in_1_pinned_);
  this->RegisterDiagnostic("output_pinned", &out_pinned_);
}


template <typename Backend>
void Merge<Backend>::WriteTestsDiagnostics(const Workspace &ws) {
  in_0_pinned_ = ws.template Input<Backend>(0).is_pinned();
  in_1_pinned_ = ws.template Input<Backend>(1).is_pinned();
  out_pinned_ = ws.template Output<Backend>(0).is_pinned();
}


DALI_SCHEMA(_conditional__Merge)
    .DocStr(R"code(Merge batch based on a predicate.)code")
    .NumInput(2)
    .NumOutput(1)
    .AddArg(
        "predicate",
        "Boolean categorization of the output batch. Must be an argument input of scalar values. "
        "Each boolean denotes if the corresponding output sample comes from the true or false "
        "branch.",
        DALI_BOOL, true)
    .SamplewisePassThrough()
    .MakeDocHidden();

DALI_REGISTER_OPERATOR(_conditional__Merge, Merge<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(_conditional__Merge, Merge<GPUBackend>, GPU);

}  // namespace dali
