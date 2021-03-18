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


#include "dali/core/math_util.h"
#include "dali/core/static_switch.h"
#include "dali/operators/numba_function/numba_func.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(NumbaFunc)
  .DocStr("") // TODO
  .NumInput(1)
  .NumOutput(1)
  .AddArg("fn_ptr", R"code(Pointer to function which should be invoked.)code", DALI_INT64);

template <typename Backend>
NumbaFunc<Backend>::NumbaFunc(const OpSpec &spec) : Base(spec) {}

template <typename Backend>
void NumbaFunc<Backend>::RunImpl(Workspace &ws) {
  auto &in = ws.template InputRef<Backend>(0);
  auto &out = ws.template OutputRef<Backend>(0);

  auto fn_ptr = this->spec_.GetArgument<uint64_t>("fn_ptr");
  auto fn = std::function<void(void*, const void*, long)>(fn_ptr);
  fn(out.raw_mutable_data(), in.raw_mutable_data(), in.size());
}

DALI_REGISTER_OPERATOR(NumbaFunc, NumbaFunc<CPUBackend>, CPU);

} // namespace dali