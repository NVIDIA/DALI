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

#ifndef DALI_KERNELS_AUDIO_FFT_FFT_CPU_H_
#define DALI_KERNELS_AUDIO_FFT_FFT_CPU_H_

#include "dali/core/format.h"
#include "dali/core/common.h"
#include "dali/core/util.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include <ffts/ffts.h>

namespace dali {
namespace kernels {
namespace audio {
namespace fft {

enum FftTransformType {
  FFT_TRANSFORM_FORWARD = 0,
  FFT_TRANSFORM_INVERSE = 1
};

enum FftOutputType {
  FFT_OUTPUT_TYPE_COMPLEX = 0,
  FFT_OUTPUT_TYPE_MAGNITUDE_ONLY = 1,
};

struct FftArgs {
  FftTransformType transform_type = FFT_TRANSFORM_FORWARD;
  FftOutputType output_type = FFT_OUTPUT_TYPE_COMPLEX;
};

template <typename OutputType, typename InputType = OutputType>
class DLL_PUBLIC FftCpu {
public:
  static_assert(std::is_same<InputType, OutputType>::value
             && std::is_same<InputType, float>::value,
    "Data types other than float are not yet supported");

  DLL_PUBLIC FftCpu() = default;

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InTensorCPU<InputType, 2> &in,
                                      const FftArgs &args) {
    DALI_ENFORCE(in.shape.size() == 2,
      "Expected a 2D tensor where first dimension represent sample id and one channel id");
    KernelRequirements req;
    auto out_shape = in.shape;
    auto &nchannels = out_shape[out_shape.size()-1];
    if (args.transform_type == FFT_TRANSFORM_FORWARD) {
      if (args.output_type == FFT_OUTPUT_TYPE_COMPLEX) {
        nchannels *= 2;  // from real to complex
      }
    } else if (args.transform_type == FFT_TRANSFORM_INVERSE) {
      DALI_ENFORCE(nchannels % 2 == 0, "Inverse FFT requires a complex input");
      if (args.output_type == FFT_OUTPUT_TYPE_MAGNITUDE_ONLY) {
        nchannels /= 2;  // from complex to real
      }
    }

    req.output_shapes = {TensorListShape<DynamicDimensions>({out_shape})};

    ScratchpadEstimator se;
    auto n = in.shape[0];
    se.add<float>(AllocType::Host, n,   32);  // N real numbers -> N floats
    se.add<float>(AllocType::Host, n+2, 32);  // (N/2)+1 complex numbers -> N+2 floats
    req.scratch_sizes = se.sizes;

    return req;
  }

  DLL_PUBLIC void Run(KernelContext &context,
                      OutTensorCPU<OutputType, 2> &out,
                      const InTensorCPU<InputType, 2> &in,
                      const FftArgs &args) {
    auto n = in.shape[0];

    DALI_ENFORCE(args.output_type == FFT_OUTPUT_TYPE_COMPLEX,
      "Output format other than complex data is not yet supported");

    int sign = (args.transform_type == FFT_TRANSFORM_FORWARD) ? FFTS_FORWARD : FFTS_BACKWARD;
    if (!plan_ || n_ != n || sign_ != sign) {
      plan_= FftsPlanPtr(ffts_init_1d_real(n, sign), ffts_free);
      n_ = n;
      sign_ = sign;
      DALI_ENFORCE(plan_ != nullptr, "Could not initialize ffts plan");
    }

    float* in_buf = context.scratchpad->template Allocate<float>(AllocType::Host, n, 32);
    memcpy(in_buf, in.data, n*sizeof(float));

    float* out_buf = context.scratchpad->template Allocate<float>(AllocType::Host, n+2, 32);
    memset(out_buf, 0.0f, n+2);

    ffts_execute(plan_.get(), in_buf, out_buf);

    // Copying first half of the spectrum
    memcpy(out.data, out_buf, (n+2)*sizeof(float));

    // Reconstructing the second half of the spectrum
    for (int i = n/2+1; i < n; i++) {
      out.data[2*i+0] =  out_buf[2*(n-i)+0];
      out.data[2*i+1] = -out_buf[2*(n-i)+1];
    }
  }

 private:
  using FftsPlanPtr = std::unique_ptr<ffts_plan_t, decltype(&ffts_free)>;
  FftsPlanPtr plan_{nullptr, ffts_free};
  int64_t n_ = -1;;
  int sign_ = FFTS_FORWARD;
};


}  // namespace fft
}  // namespace audio
}  // namespace kernels
}  // namespace dali

  #endif  // DALI_KERNELS_AUDIO_FFT_FFT_CPU_H_