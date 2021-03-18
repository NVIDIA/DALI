// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


#include "dali/operators/numba_function/numba_func.h"

namespace dali {

DALI_SCHEMA(NumbaFunc)
  .DocStr("")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("fn_ptr", R"code(Pointer to function which should be invoked.)code", DALI_INT64);

template <typename Backend>
NumbaFunc<Backend>::NumbaFunc(const OpSpec &spec) : Base(spec) {}

template <>
void NumbaFunc<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  const auto &in = ws.Input<CPUBackend>(0);
  auto &out = ws.Output<CPUBackend>(0);

  auto fn_ptr = this->spec_.GetArgument<uint64_t>("fn_ptr");
  ((void (*)(void*, const void*, uint64_t))fn_ptr)(
    out.raw_mutable_data(), in.raw_data(), in.size());
}

DALI_REGISTER_OPERATOR(NumbaFunc, NumbaFunc<CPUBackend>, CPU);

}  // namespace dali
