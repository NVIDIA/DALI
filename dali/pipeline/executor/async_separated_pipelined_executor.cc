// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/executor/async_separated_pipelined_executor.h"

namespace dali {

void AsyncSeparatedPipelinedExecutor::RunCPU() {
  CheckForErrors();
  // Run the cpu work. We schedule it in queue and wait for free buffers
  cpu_thread_.DoWork([this]() { SeparatedPipelinedExecutor::RunCPU(); });
}

void AsyncSeparatedPipelinedExecutor::RunMixed() {
  CheckForErrors();
  mixed_thread_.DoWork([this]() { SeparatedPipelinedExecutor::RunMixed(); });
}

void AsyncSeparatedPipelinedExecutor::RunGPU() {
  CheckForErrors();
  gpu_thread_.DoWork([this]() { SeparatedPipelinedExecutor::RunGPU(); });
}

}  // namespace dali
