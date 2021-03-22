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

DALI_SCHEMA(experimental__NumbaFunc)
  .DocStr(R"code(Invokes a compiled Numba function passed as a pointer.
.. note::
    This operator is experimental and its API might change without notice.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("fn_ptr", R"code(Numba function pointer.
  
The function should be a Numba C callback function (annotated with cfunc) with the following function signature
<insert expected function signature>.
)code", DALI_INT64)
  .AddOptionalArg<int>("setup_fn", R"code(Pointer to function which should return output shape.)code", DALI_INT64);

template <typename Backend>
NumbaFunc<Backend>::NumbaFunc(const OpSpec &spec) : Base(spec) {
  fn_ptr_ = spec.GetArgument<uint64_t>("fn_ptr");
  setup_fn_ = spec.GetArgument<uint64_t>("setup_fn");
}

template<>
void NumbaFunc<CPUBackend>::RunUserSetupFunc(std::vector<OutputDesc> &output_desc,
    const workspace_t<CPUBackend> &ws) {
  const auto &in = ws.InputRef<CPUBackend>(0);
  if (!setup_fn_) {
    output_desc[0] = {in.shape(), in.type()};
    return;
  }

  auto in_shape = in.shape();
  auto N = in_shape.num_samples();
  auto ndim = in_shape.sample_dim();
  TensorListShape<> output_shape;
  output_shape.resize(N, ndim);
  DALIDataType out_type = DALIDataType::DALI_NO_TYPE;
  DALIDataType in_type = in.type().id();
  ((void (*)(void*, const void*, int32_t, int32_t, void*, const void*, int64_t, int64_t))setup_fn_)(
    output_shape.tensor_shape_span(0).begin(), in_shape.tensor_shape_span(0).begin(),
    N, ndim, &out_type, &in_type, 1, 1);

  DALI_ENFORCE(out_type != DALIDataType::DALI_NO_TYPE, "Output type was not set by the custom setup function.");
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < ndim; d++) {
      DALI_ENFORCE(output_shape.tensor_shape_span(i)[d] >= 0,
        make_string(d, "-th", " dimension of ", i, "-th sample's shape is negative."));
    }
  }
  output_desc[0].type = dali::TypeTable::GetTypeInfo(out_type);
  output_desc[0].shape = output_shape;
}

template <>
bool NumbaFunc<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
    const workspace_t<CPUBackend> &ws) {
  const auto &in = ws.InputRef<CPUBackend>(0);
  output_desc.resize(1);
  RunUserSetupFunc(output_desc, ws);
  return true;
}

template <>
void NumbaFunc<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &in = ws.InputRef<CPUBackend>(0);
  auto in_shape = in.shape();
  auto &out = ws.OutputRef<CPUBackend>(0);
  auto out_shape = out.shape();
  auto& tp = ws.GetThreadPool();

  for (int sample_id = 0; sample_id < in_shape.num_samples(); sample_id++) {
    tp.AddWork([&, fn_ptr = fn_ptr_, sample_id](int thread_id) {
      ((void (*)(void*, const void*, const void*, const void*, uint64_t))fn_ptr)(
        out[sample_id].raw_mutable_data(), in[sample_id].raw_data(),
        out_shape.tensor_shape_span(sample_id).data(),
        in_shape.tensor_shape_span(sample_id).data(),
        out_shape.sample_dim());
    }, out_shape.tensor_size(sample_id));
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(experimental__NumbaFunc, NumbaFunc<CPUBackend>, CPU);

}  // namespace dali
