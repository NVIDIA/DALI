// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/test/plugins/dummy/dummy.h"

namespace other_ns {

template<>
void Dummy<::dali::CPUBackend>::RunImpl(::dali::SampleWorkspace &ws) {
  auto &input = ws.Input<::dali::CPUBackend>(0);
  auto &output = ws.Output<::dali::CPUBackend>(0);
  output.set_type(input.type());
  output.ResizeLike(input);

  ::dali::TypeInfo type = input.type();
  type.Copy<::dali::CPUBackend, ::dali::CPUBackend>(
      output.raw_mutable_data(),
      input.raw_data(), input.size(), 0);
}

}  // namespace other_ns

DALI_REGISTER_OPERATOR(CustomDummy, ::other_ns::Dummy<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(CustomDummy)
  .DocStr("Make a copy of the input tensor")
  .NumInput(1)
  .NumOutput(1);


