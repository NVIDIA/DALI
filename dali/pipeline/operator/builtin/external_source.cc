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

#include "dali/pipeline/operator/builtin/external_source.h"
#include <functional>
#include <list>

namespace dali {

template <>
void ExternalSource<CPUBackend>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  auto &thread_pool = ws.GetThreadPool();
  ForwardCurrentData(output, thread_pool);
}


template<>
void ExternalSource<GPUBackend>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<GPUBackend>(0);
  cudaStream_t stream_used = ws.has_stream() ? ws.stream() : 0;
  ForwardCurrentData(output, stream_used);
}


DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<GPUBackend>, GPU);

}  // namespace dali
