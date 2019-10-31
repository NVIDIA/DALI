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

#include "dali/kernels/audio/fft/fft_cpu.h"
#include "dali/core/format.h"
#include "dali/core/common.h"
#include "dali/core/util.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include <ffts/ffts.h>

namespace dali {
namespace kernels {
namespace audio {
namespace fft {

namespace {

inline bool is_pow2(int64_t n) {
  return (n & (n-1)) == 0;
}

inline bool can_use_real_impl(int64_t n) {
  // Real impl can be selected when doing forward transform and n is a power of 2
  return is_pow2(n);
}

inline int64_t size_in_buf(int64_t n) {
  // Real impl input needs:    N real numbers    -> N floats
  // Complex impl input needs: N complex numbers -> 2*N floats
  return can_use_real_impl(n) ? n : 2*n;
}

inline int64_t size_out_buf(int64_t n) {
  // Real impl output needs:    (N/2)+1 complex numbers -> N+2 floats
  // Complex impl output needs: N complex numbers       -> 2*N floats
  return can_use_real_impl(n) ? n+2 : 2*n;
}

}  // namespace

template <typename OutputType, typename InputType, int Dims>
KernelRequirements Fft1DCpu<OutputType, InputType, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<InputType, Dims> &in,
    const FftArgs &args) {
  ValidateArgs(args);

  KernelRequirements req;
  auto out_shape = in.shape;
  if (args.output_type == FFT_OUTPUT_TYPE_COMPLEX) {
    out_shape[args.channel_axis] *= 2;  // from real to complex
  }
  req.output_shapes = {TensorListShape<DynamicDimensions>({out_shape})};

  ScratchpadEstimator se;
  auto n = in.shape[args.transform_axis];
  se.add<float>(AllocType::Host, size_in_buf(n),  32);
  se.add<float>(AllocType::Host, size_out_buf(n), 32);
  req.scratch_sizes = se.sizes;

  return req;
}

template <typename OutputType, typename InputType, int Dims>
void Fft1DCpu<OutputType, InputType, Dims>::Run(
    KernelContext &context,
    OutTensorCPU<OutputType, Dims> &out,
    const InTensorCPU<InputType, Dims> &in,
    const FftArgs &args) {
  ValidateArgs(args);
  auto n = in.shape[args.transform_axis];

  DALI_ENFORCE(args.output_type == FFT_OUTPUT_TYPE_COMPLEX,
    "Output format other than complex data is not yet supported");

  bool use_real_impl = can_use_real_impl(n);

  if (!plan_ || plan_n_ != n) {
    if (use_real_impl) {
      plan_ = {ffts_init_1d_real(n, FFTS_FORWARD), ffts_free};
    } else {
      plan_ = {ffts_init_1d(n, FFTS_FORWARD), ffts_free};
    }
    DALI_ENFORCE(plan_ != nullptr, "Could not initialize ffts plan");
    plan_n_ = n;
  }

  auto in_buf_sz = size_in_buf(n);
  // ffts requires 32-byte aligned memory
  float* in_buf = context.scratchpad->template Allocate<float>(AllocType::Host, in_buf_sz, 32);

  auto out_buf_sz = size_out_buf(n);
  // ffts requires 32-byte aligned memory
  float* out_buf = context.scratchpad->template Allocate<float>(AllocType::Host, out_buf_sz, 32);

  const InputType* in_data = in.data;
  OutputType* out_data = out.data;

  if (use_real_impl) {
    for (int i = 0; i < n; i++) {
      in_buf[i] = ConvertSat<float>(in_data[i]);
    }
  } else {
    for (int i = 0; i < n; i++) {
      in_buf[2*i] = ConvertSat<float>(in_data[i]);
      in_buf[2*i+1] = 0.0f;
    }
  }

  ffts_execute(plan_.get(), in_buf, out_buf);

  // For complex impl, out_buf_sz contains the whole spectrum,
  // for real impl, the second half of the spectrum is ommited and should be constructed from the
  // first half
  for (int i = 0; i < out_buf_sz; i++) {
    out_data[i] = ConvertSat<OutputType>(out_buf[i]);
  }
  if (use_real_impl) {
    // Reconstructing the second half of the spectrum
    for (int i = n/2+1; i < n; i++) {
      out_data[2*i+0] = ConvertSat<OutputType>( out_buf[2*(n-i)+0]);
      out_data[2*i+1] = ConvertSat<OutputType>(-out_buf[2*(n-i)+1]);
    }
  }
}

template <typename OutputType, typename InputType, int Dims>
void Fft1DCpu<OutputType, InputType, Dims>::ValidateArgs(const FftArgs& args) {
  DALI_ENFORCE(args.channel_axis >= 0 && args.channel_axis < Dims,
    make_string("Channel axis ", args.channel_axis, " is out of bounds [0, ", Dims, ")"));
  DALI_ENFORCE(args.transform_axis >= 0 && args.transform_axis < Dims,
    make_string("Transform axis ", args.transform_axis, " is out of bounds [0, ", Dims, ")"));
  DALI_ENFORCE(args.transform_axis == 0 && args.channel_axis == 1,
    "Expected 2D data where dim 0 represents the sample space and dim 1 the different channels");
}

template class Fft1DCpu<float, float, 2>;

}  // namespace fft
}  // namespace audio
}  // namespace kernels
}  // namespace dali
