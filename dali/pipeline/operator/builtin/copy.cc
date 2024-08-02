// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/builtin/copy.h"

namespace dali {

template <>
void Copy<CPUBackend, CPUBackend>::RunImpl(Workspace &ws) {
  auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);

  int batch_size = input.num_samples();
  output.SetLayout(input.GetLayout());
  auto shapes = input.shape();

  auto &thread_pool = ws.GetThreadPool();
  for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
    thread_pool.AddWork(
        [sample_id, &input, &output](int tid) {
          output.CopySample(sample_id, input, sample_id, AccessOrder::host());
        },
        shapes.tensor_size(sample_id));
  }
  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(Copy, Copy<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Copy, Copy<GPUBackend>, GPU);

using CopyD2H = Copy<CPUBackend, GPUBackend>;
DALI_REGISTER_OPERATOR(CopyD2H, CopyD2H, CPU);

DALI_SCHEMA(Copy)
  .DocStr("Creates a copy of the input tensor.")
  .NumInput(1)
  .NumOutput(1)
  .AllowSequences()
  .SupportVolumetric();


DALI_SCHEMA(CopyD2H)
  .DocStr("Creates a copy of the input tensor.")
  .NumInput(1)
  .InputDevice(0, InputDevice::GPU)
  .NumOutput(1)
  .AllowSequences()
  .SupportVolumetric();


}  // namespace dali
