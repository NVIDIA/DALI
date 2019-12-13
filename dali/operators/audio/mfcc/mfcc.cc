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

#include "dali/operators/audio/mfcc/mfcc.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/signal/dct/dct_cpu.h"
#include "dali/pipeline/data/views.h"

#define MFCC_SUPPORTED_TYPES (float)
#define MFCC_SUPPORTED_NDIMS (2, 3, 4)

static constexpr int kNumInputs = 1;
static constexpr int kNumOutputs = 1;

namespace dali {

DALI_SCHEMA(MFCC)
    .DocStr(R"code(Mel Frequency Cepstral Coefficiencs (MFCC).
Computes MFCCs from a mel spectrogram)code")
    .NumInput(kNumInputs)
    .NumOutput(kNumOutputs)
    .AddOptionalArg("n_mfcc",
      R"code(Number of MFCC coefficients)code",
      20)
    .AddOptionalArg("dct_type",
      R"code(Discrete Cosine Transform type. Supported types are: 1, 2, 3, 4.
The formulas used to calculate the DCT are equivalent to those described in
https://en.wikipedia.org/wiki/Discrete_cosine_transform)code",
      2)
    .AddOptionalArg("normalize",
      R"code(If true, the DCT will use an ortho-normal basis.
Note: Normalization is not supported for `dct_type=1`.)code",
      false)
    .AddOptionalArg("axis",
      R"code(Axis over which the transform will be applied.
If not provided, the outer-most dimension will be used.)code",
      0)
    .AddOptionalArg("lifter",
      R"code(Cepstral filtering (also known as `liftering`) coefficient.
If `lifter>0`, the MFCCs will be scaled according to the following formula::

  MFFC[i] = MFCC[i] * (1 + sin(pi * (i + 1) / lifter)) * (lifter / 2)

)code",
      0.0f);

template <>
bool MFCC<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                 const workspace_t<CPUBackend> &ws) {
  output_desc.resize(kNumOutputs);
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  kernels::KernelContext ctx;
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto nthreads = ws.GetThreadPool().size();

  TYPE_SWITCH(input.type().id(), type2id, T, MFCC_SUPPORTED_TYPES, (
    VALUE_SWITCH(in_shape.sample_dim(), Dims, MFCC_SUPPORTED_NDIMS, (
      using DctKernel = kernels::signal::dct::Dct1DCpu<T, T, Dims>;
      kmgr_.Initialize<DctKernel>();
      kmgr_.Resize<DctKernel>(nthreads, nsamples);
      output_desc[0].type = TypeInfo::Create<T>();
      output_desc[0].shape.resize(nsamples, Dims);
      for (int i = 0; i < nsamples; i++) {
        const auto in_view = view<const T, Dims>(input[i]);
        auto &req = kmgr_.Setup<DctKernel>(i, ctx, in_view, args_);
        output_desc[0].shape.set_tensor_shape(i, req.output_shapes[0][0].shape);
      }
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT
  return true;
}

template <>
void MFCC<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto& thread_pool = ws.GetThreadPool();

  TYPE_SWITCH(input.type().id(), type2id, T, MFCC_SUPPORTED_TYPES, (
    VALUE_SWITCH(in_shape.sample_dim(), Dims, MFCC_SUPPORTED_NDIMS, (
      using DctKernel = kernels::signal::dct::Dct1DCpu<T, T, Dims>;
      for (int i = 0; i < input.shape().num_samples(); i++) {
        thread_pool.DoWorkWithID(
          [this, &input, &output, i](int thread_id) {
            kernels::KernelContext ctx;
            auto in_view = view<const T, Dims>(input[i]);
            auto out_view = view<T, Dims>(output[i]);
            kmgr_.Run<DctKernel>(thread_id, i, ctx, out_view, in_view, args_);
          });
      }
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT

  thread_pool.WaitForWork();
}

DALI_REGISTER_OPERATOR(MFCC, MFCC<CPUBackend>, CPU);

}  // namespace dali
