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

#include "dali/operators/signal/decibel/to_decibels_op.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/signal/decibel/to_decibels_cpu.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(ToDecibels)
    .DocStr(R"code(Converts a magnitude (real, positive) to the decibel scale by using
the following formula::

  min_ratio = pow(10, cutoff_db / multiplier)
  out[i] = multiplier * log10( max(min_ratio, input[i] / reference) ))code")
    .NumInput(kNumInputs)
    .NumOutput(kNumOutputs)
    .AddOptionalArg("multiplier",
      R"code(Factor by which the logarithm is multiplied. The value is typically 10.0 or 20.0,
which depends on whether the magnitude is squared.)code",
      10.0f)
    .AddOptionalArg("reference",
      R"code(Reference magnitude.

If a value is not provided, the maximum value for the input will be used as reference.

.. note::
  The maximum of the input will be calculated on a per-sample basis.
)code",
      0.0f)
    .AddOptionalArg("cutoff_db",
      R"code(Minimum or cut-off ratio in dB.

Any value below this value will saturate. For example, a value of ``cutoff_db=-80`` corresponds
to a minimum ratio of ``1e-8``.)code",
      -200.0f);

template <>
bool ToDecibels<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                       const workspace_t<CPUBackend> &ws) {
  output_desc.resize(kNumOutputs);
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  kernels::KernelContext ctx;
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto nthreads = ws.GetThreadPool().NumThreads();

  TYPE_SWITCH(input.type().id(), type2id, T, (float), (
    using ToDbKernel = kernels::signal::ToDecibelsCpu<T>;
    kmgr_.Initialize<ToDbKernel>();
    kmgr_.Resize<ToDbKernel>(nthreads, nsamples);
    output_desc[0].shape = in_shape;
    output_desc[0].type = TypeInfo::Create<T>();
    for (int i = 0; i < nsamples; i++) {
      const auto in_view = view<const T>(input[i]);
      auto &req = kmgr_.Setup<ToDbKernel>(i, ctx, in_view, args_);
    }
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT
  return true;
}

template <>
void ToDecibels<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto& thread_pool = ws.GetThreadPool();

  TYPE_SWITCH(input.type().id(), type2id, T, (float), (
    using ToDbKernel = kernels::signal::ToDecibelsCpu<T>;
    for (int i = 0; i < input.shape().num_samples(); i++) {
      thread_pool.AddWork(
        [this, &input, &output, i](int thread_id) {
          kernels::KernelContext ctx;
          auto in_view = view<const T>(input[i]);
          auto out_view = view<T>(output[i]);
          kmgr_.Run<ToDbKernel>(thread_id, i, ctx, out_view, in_view, args_);
        }, in_shape.tensor_size(i));
    }
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT

  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(ToDecibels, ToDecibels<CPUBackend>, CPU);

}  // namespace dali
