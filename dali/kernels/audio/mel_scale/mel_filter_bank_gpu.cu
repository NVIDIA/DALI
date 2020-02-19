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

#include "dali/kernels/audio/mel_scale/mel_filter_bank_gpu.h"

namespace dali {
namespace kernels {
namespace audio {

template <typename T>
__global__ void MelFilterBankKernel(T **outs, T **ins, int64_t *windows_num,
                                    int64_t in_stride,
                                    int64_t *interval_ends, T *weights_down,
                                    bool normalize, T *norm_factors,
                                    int64_t fftbin_start, int64_t fftbin_end) {
  T *in = ins[blockIdx.z];
  T *out = outs[blockIdx.z];
  int64_t nwindows = windows_num[blockIdx.z];
  int64_t mel_start = blockIdx.y * blockDim.y + threadIdx.y;
  
}

template <typename T, int Dims>
class MelFilterBankGpu<T, Dims>::Impl : MelFilterImplBase<T, Dims> {
 public:
  template <typename MelScale>
  Impl(KernelContext &ctx, MelScale mel_scale, const MelFilterBankArgs &args)
  : MelFilterImplBase<T, Dims>(mel_scale, args) {
    T mel = mel_low_ + mel_delta_;

    int64_t fftbin = fftbin_start_;
    T f = fftbin * hz_step_;

    interval_ends_.back() = fftbin_end_;
    for (int64_t interval = 0; interval < args_.nfilter - 1;
         interval++, mel += mel_delta_) {
      T freq = mel_scale.mel_to_hz(mel);
      interval_ends_[interval] = std::floor(freq / hz_step_);
    }
  }

  void Compute(int nsamples, T *outs, T *ins, int64_t nwindows,
               int64_t out_stride = -1, int64_t in_stride = -1) {

  }

 private:
  std::vector<int64_t> interval_ends_;
  USE_MEL_FILTER_IMPL_MEMBERS(T, Dims);
};

}
}
}