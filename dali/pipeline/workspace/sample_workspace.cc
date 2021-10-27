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

#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

void MakeSampleView(SampleWorkspace& sample, HostWorkspace& batch, int data_idx, int thread_idx) {
  sample.Clear();
  sample.set_data_idx(data_idx);
  sample.set_thread_idx(thread_idx);
  int num_inputs = batch.NumInput();
  for (int i = 0; i < num_inputs; i++) {
    if (batch.InputIsType<CPUBackend>(i)) {
      auto &input_ref = batch.UnsafeMutableInput<CPUBackend>(i);
      sample.AddInput(&input_ref[data_idx]);
    } else {
      auto &input_ref = batch.UnsafeMutableInput<GPUBackend>(i);
      sample.AddInput(&input_ref[data_idx]);
    }
  }

  int num_outputs = batch.NumOutput();
  for (int i = 0; i < num_outputs; i++) {
    if (batch.OutputIsType<CPUBackend>(i)) {
      auto &output_ref = batch.Output<CPUBackend>(i);
      sample.AddOutput(&output_ref[data_idx]);
    } else {
      auto &output_ref = batch.Output<GPUBackend>(i);
      sample.AddOutput(&output_ref[data_idx]);
    }
  }
  for (auto& arg_pair : batch) {
    assert(!arg_pair.second.should_update);
    sample.AddArgumentInput(arg_pair.first, arg_pair.second.tvec);
  }
}

}  // namespace dali
