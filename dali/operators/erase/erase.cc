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

#include "dali/operators/erase/erase.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(Erase)
  .DocStr(R"code(TODO....)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg<float>("anchor",
    R"code()code",
    vector<float>(), true)
  .AddOptionalArg<float>("shape",
    R"code()code",
    vector<float>(), true)
  .AddOptionalArg("axes",
    R"code(Order of dimensions used for anchor and shape arguments, as dimension indexes)code",
    std::vector<int>{0, 1})
  .AddOptionalArg("axis_names",
    R"code(Order of dimensions used for anchor and shape arguments, as described in layout.
If provided, `axis_names` takes higher priority than `axes`)code",
    "HW")
  .AllowSequences()
  .SupportVolumetric();

template <>
void Erase<CPUBackend>::RunImpl(HostWorkspace &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto& thread_pool = ws.GetThreadPool();

  // output.SetLayout(layout);
}

DALI_REGISTER_OPERATOR(Erase, Erase<CPUBackend>, CPU);

}  // namespace dali
