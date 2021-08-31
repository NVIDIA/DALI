// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/webdataset_reader_op.h"
#include <algorithm>
#include <string>

namespace dali {

void WebdatasetReader::RunImpl(HostWorkspace& ws) {
  int num_outputs = ws.NumOutput();
  int num_samples = GetCurrBatchSize();

  for (int data_idx = 0; data_idx < num_samples; data_idx++) {
    auto& sample = GetSample(data_idx);
    for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
      ws.OutputRef<CPUBackend>(output_idx)[data_idx].ShareData(&sample[output_idx]);
    }
  }
}

DALI_SCHEMA(readers__Webdataset)
    .DocStr(
        R"code(
          To be filled in
        )code")
    .NumInput(0)
    .OutputFn([](const OpSpec& spec) {
      return spec.GetRepeatedArgument<std::string>("ext").size();
    })
    .AddArg("uris", R"code(To be filled in)code", DALI_STRING_VEC)
    .AddArg("configs", R"code(To be filled in)code", DALI_STRING_VEC)
    .AddArg("ext", R"code(To be filled in)code", DALI_STRING_VEC)
    .AddOptionalArg("missing_component_behavior", R"code(To be filled in)code", "")
    .AddOptionalArg("dtype", R"code(To be filled in: numeric)code", DALI_DATA_TYPE_VEC,
                    nullptr)  // default is a vector of uint8
    .AddParent("LoaderBase");

DALI_REGISTER_OPERATOR(readers__Webdataset, WebdatasetReader, CPU);

}  // namespace dali
