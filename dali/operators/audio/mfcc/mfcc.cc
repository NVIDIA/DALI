// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/common/utils.h"
#include "dali/kernels/common/for_axis.h"
#include "dali/pipeline/data/views.h"

#define MFCC_SUPPORTED_NDIMS (2, 3, 4)

static constexpr int kNumInputs = 1;
static constexpr int kNumOutputs = 1;

namespace dali {

namespace detail {

template <>
DLL_PUBLIC void LifterCoeffs<CPUBackend>::Calculate(int64_t target_length, float lifter,
                                                    cudaStream_t)  {
  // If different lifter argument, clear previous coefficients
  if (lifter_ != lifter) {
    coeffs_.clear();
    lifter_ = lifter;
  }

  // 0 means no liftering
  if (lifter_ == 0.0f)
    return;

  // Calculate remaining coefficients (if necessary)
  if (static_cast<int64_t>(coeffs_.size()) < target_length) {
    int64_t start = coeffs_.size(), end = target_length;
    coeffs_.resize(target_length);
    CalculateCoeffs(coeffs_.data() + start, start, target_length - start);
  }
}


template <typename T, int Dims>
void ApplyLifter(const kernels::OutTensorCPU<T, Dims> &inout, int axis, const T* lifter_coeffs) {
  assert(axis >= 0 && axis < Dims);
  assert(lifter_coeffs != nullptr);
  auto* data = inout.data;
  auto shape = inout.shape;
  auto strides = kernels::GetStrides(shape);
  kernels::ForAxis(
    data, data, shape.data(), strides.data(), shape.data(), strides.data(), axis, Dims,
    [lifter_coeffs](
      T *out_data, const T *in_data, int64_t out_size, int64_t out_stride,
      int64_t in_size, int64_t in_stride) {
        int64_t idx = 0;
        assert(out_size == in_size);
        assert(out_stride == in_stride);
        for (int64_t k = 0; k < out_size; k++, idx += out_stride) {
          out_data[idx] = lifter_coeffs[k] * in_data[idx];
        }
      });
}

}  // namespace detail

DALI_SCHEMA(MFCC)
    .DocStr(R"code(Computes Mel Frequency Cepstral Coefficiencs (MFCC) from
a mel spectrogram.)code")
    .NumInput(kNumInputs)
    .NumOutput(kNumOutputs)
    .AddOptionalArg("n_mfcc",
      R"code(Number of MFCC coefficients.)code",
      20)
    .AddOptionalArg("dct_type",
      R"code(Discrete Cosine Transform type.

The supported types are 1, 2, 3, 4. The formulas that are used to calculate the DCT are equivalent
to those described in https://en.wikipedia.org/wiki/Discrete_cosine_transform (the numbers
correspond to types listed in
https://en.wikipedia.org/wiki/Discrete_cosine_transform#Formal_definition).)code",
      2)
    .AddOptionalArg("normalize",
      R"code(If set to True, the DCT uses an ortho-normal basis.

.. note::
  Normalization is not supported when dct_type=1.)code",
      false)
    .AddOptionalArg("axis",
      R"code(Axis over which the transform will be applied.

If a value is not provided, the outer-most dimension will be used.)code",
      0)
    .AddOptionalArg("lifter",
      R"code(Cepstral filtering coefficient, which is also known as the liftering coefficient.

If the lifter coefficient is greater than 0, the MFCCs will be scaled based on
the following formula::

    MFFC[i] = MFCC[i] * (1 + sin(pi * (i + 1) / lifter)) * (lifter / 2)
)code",
      0.0f);

template <>
bool MFCC<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                 const workspace_t<CPUBackend> &ws) {
  GetArguments(ws);
  output_desc.resize(kNumOutputs);
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  kernels::KernelContext ctx;
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto nthreads = ws.GetThreadPool().NumThreads();

  int64_t max_length = -1;

  int ndim = in_shape.sample_dim();
  DALI_ENFORCE(axis_ >= 0 && axis_ < ndim,
               make_string("Axis ", axis_, " is out of bounds [0,", ndim, ")"));

  TYPE_SWITCH(input.type().id(), type2id, T, MFCC_SUPPORTED_TYPES, (
    VALUE_SWITCH(in_shape.sample_dim(), Dims, MFCC_SUPPORTED_NDIMS, (
      using DctKernel = kernels::signal::dct::Dct1DCpu<T, T, Dims>;
      kmgr_.Initialize<DctKernel>();
      kmgr_.Resize<DctKernel>(nthreads, nsamples);
      output_desc[0].type = TypeInfo::Create<T>();
      output_desc[0].shape.resize(nsamples, Dims);
      for (int i = 0; i < nsamples; i++) {
        const auto in_view = view<const T, Dims>(input[i]);
        auto &req = kmgr_.Setup<DctKernel>(i, ctx, in_view, args_[i], axis_);
        auto out_shape = req.output_shapes[0][0];
        output_desc[0].shape.set_tensor_shape(i, out_shape);
        if (out_shape[axis_] > max_length) {
          max_length = out_shape[axis_];
        }
      }
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT

  lifter_coeffs_.Calculate(max_length, lifter_);
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
        thread_pool.AddWork(
          [this, &input, &output, i](int thread_id) {
            kernels::KernelContext ctx;
            auto in_view = view<const T, Dims>(input[i]);
            auto out_view = view<T, Dims>(output[i]);
            kmgr_.Run<DctKernel>(thread_id, i, ctx, out_view, in_view, args_[i], axis_);
            if (lifter_ != 0.0f) {
              assert(static_cast<int64_t>(lifter_coeffs_.size()) >= out_view.shape[axis_]);
              detail::ApplyLifter(out_view, axis_, lifter_coeffs_.data());
            }
          }, in_shape.tensor_size(i));
      }
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT

  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(MFCC, MFCC<CPUBackend>, CPU);

}  // namespace dali
