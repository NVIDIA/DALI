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
  .DocStr(R"code(Invokes numba function passes as ``fn_ptr``.
  
This feature is experimental. Note that API for it may change in future.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("fn_ptr", R"code(Pointer to function which should be invoked.)code", DALI_INT64);

template <typename Backend>
NumbaFunc<Backend>::NumbaFunc(const OpSpec &spec) : Base(spec) {
  fn_ptr_ = spec.GetArgument<uint64_t>("fn_ptr");
}

template <>
void NumbaFunc<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &in = ws.InputRef<CPUBackend>(0);
  auto &out = ws.OutputRef<CPUBackend>(0);
  auto out_shape = out.shape();
  auto& tp = ws.GetThreadPool();

  for (int sample_id = 0; sample_id < in.shape().num_samples(); sample_id++) {
    tp.AddWork([&, fn_ptr = fn_ptr_, sample_id](int thread_id) {
      ((void (*)(void*, const void*, uint64_t))fn_ptr)(
        out[sample_id].raw_mutable_data(), in[sample_id].raw_data(), in[sample_id].size());
    }, out_shape.tensor_size(sample_id));
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(NumbaFunc, NumbaFunc<CPUBackend>, CPU);

}  // namespace dali
