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

#include "dali/kernels/audio/mel_scale/mel_filter_bank_cpu.h"
#include <cmath>
#include <complex>
#include <algorithm>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/for_axis.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/audio/mel_scale/mel_scale.h"

namespace dali {
namespace kernels {
namespace audio {

// In the outer loop we travel at a linearly spaced frequency grid in the mel scale
// Each triangular filter is defined by three points in this grid (left, center, right)
// For each iteration we process a range between two mel frequencies in the grid, calculating
// the contribution of each FFT bin to 2 triangular filters (one is in the negative slope region
// and the other in the positive slope region), except for the first and last iteration.
// In total, we do a single pass on every FFT bin column
//
// For every FFT bin we compute the weight for each filter and travel through the row, computing
// the contributions on every window of the spectrogram (horizontal axis)
//
template <typename T, int Dims>
class MelFilterBankCpu<T, Dims>::Impl: public MelFilterImplBase<T, Dims> {
 public:
  template <typename MelScale>
  Impl(MelScale mel_scale, const MelFilterBankArgs &args)
      : MelFilterImplBase<T, Dims>(mel_scale, args) {
    double mel = mel_low_ + mel_delta_;
    auto nfilter = args.nfilter;
    assert(args.axis == Dims - 1 || args.axis == Dims - 2);
    if (args.axis == Dims - 2) {
      intervals_.resize(fftbin_size_, -1);
      int fftbin = fftbin_start_;
      double f = fftbin * hz_step_;
      for (int interval = 0; interval < nfilter + 1; interval++, mel += mel_delta_) {
        double freq = mel_scale.mel_to_hz(interval == nfilter ? mel_high_ : mel);
        for (; fftbin <= fftbin_end_ && f < freq; fftbin++, f = fftbin * hz_step_) {
          intervals_[fftbin] = interval;
        }
      }
    } else {  // args.axis == Dims - 1
      interval_ends_.resize(nfilter + 2);
      interval_ends_[0] = fftbin_start_;
      interval_ends_[nfilter + 1] = fftbin_end_ + 1;
      for (int interval = 1; interval < nfilter + 1; interval++, mel += mel_delta_) {
        double freq = mel_scale.mel_to_hz(mel);
        interval_ends_[interval] = std::ceil(freq / hz_step_);
      }
    }
  }

  void ComputeFreqMajor(T* out, const T* in, int64_t nwindows) {
    int nfilter = args_.nfilter;

    std::memset(out, 0, sizeof(T) * nfilter * nwindows);
    for (int64_t fftbin = fftbin_start_; fftbin <= fftbin_end_; fftbin++) {
      auto *in_row_start = in + fftbin * nwindows;
      auto filter_up = intervals_[fftbin];
      auto weight_up = T(1) - weights_down_[fftbin];
      auto filter_down = filter_up - 1;
      auto weight_down = weights_down_[fftbin];

      if (filter_down >= 0) {
        if (args_.normalize)
          weight_down *= norm_factors_[filter_down];
        auto *out_row_start = out + filter_down * nwindows;
        for (int t = 0; t < nwindows; t++) {
          out_row_start[t] += weight_down * in_row_start[t];
        }
      }

      if (filter_up >= 0 && filter_up < nfilter) {
        if (args_.normalize)
          weight_up *= norm_factors_[filter_up];
        auto *out_row_start = out + filter_up * nwindows;
        for (int t = 0; t < nwindows; t++) {
          out_row_start[t] += weight_up * in_row_start[t];
        }
      }
    }
  }

  void ComputeTimeMajor(T* out, const T* in, int64_t nwindows) {
    int nfilter = args_.nfilter;
    for (int t = 0; t < nwindows; t++) {
      const T *in_row = in + t * fftbin_size_;
      for (int m = 0; m < nfilter; m++) {
        T val = 0;
        int fftbin = interval_ends_[m];
        int f1 = interval_ends_[m + 1];
        int f2 = interval_ends_[m + 2];
        for (; fftbin < f1; ++fftbin) {
          auto weight_up = T(1) - weights_down_[fftbin];
          if (args_.normalize)
            weight_up *= norm_factors_[m];
          val += in_row[fftbin] * weight_up;
        }
        for (; fftbin < f2; ++fftbin) {
          auto weight_down = weights_down_[fftbin];
          if (args_.normalize)
            weight_down *= norm_factors_[m];
          val += in_row[fftbin] * weight_down;
        }
        *out++ = val;
      }
    }
  }

 private:
  std::vector<int> intervals_;
  std::vector<int> interval_ends_;
  USE_MEL_FILTER_IMPL_MEMBERS(T, Dims);
};

template <typename T, int Dims>
MelFilterBankCpu<T, Dims>::MelFilterBankCpu() = default;

template <typename T, int Dims>
MelFilterBankCpu<T, Dims>::~MelFilterBankCpu() = default;

template <typename T, int Dims>
KernelRequirements MelFilterBankCpu<T, Dims>::Setup(KernelContext &context,
                                                    const InTensorCPU<T, Dims> &in,
                                                    const MelFilterBankArgs &orig_args) {
  auto args = orig_args;
  args.axis = args.axis >= 0 ? args.axis : Dims - 2;
  DALI_ENFORCE(args.axis == Dims - 2 || args.axis == Dims - 1,
               "Input is expected to be a spectrogram with the last two dimensions being "
               "(fftbin_idx, frame_idx), frequency major, or (frame_idx, fftbin_idx), time major.");
  auto out_shape = in.shape;
  out_shape[args.axis] = args.nfilter;

  std::vector<TensorShape<DynamicDimensions>> tmp = {out_shape};  // workaround for clang-6 bug
  KernelRequirements req;
  req.output_shapes = {TensorListShape<DynamicDimensions>(tmp)};

  args.nfft = args.nfft > 0 ? args.nfft : 2 * (in.shape[args.axis] - 1);
  args.freq_high = args.freq_high > 0 ? args.freq_high : args.sample_rate / 2;
  if (!impl_ || impl_->Args() != args) {
    impl_.reset();
    switch (args.mel_formula) {
      case MelScaleFormula::HTK:
        impl_ = std::make_unique<Impl>(HtkMelScale<T>(), args);
        break;
      case MelScaleFormula::Slaney:
      default:
        impl_ = std::make_unique<Impl>(SlaneyMelScale<T>(), args);
        break;
    }
  }
  return req;
}

template <typename T, int Dims>
void MelFilterBankCpu<T, Dims>::Run(KernelContext &context, const OutTensorCPU<T, Dims> &out,
                                    const InTensorCPU<T, Dims> &in) {
  DALI_ENFORCE(impl_ != nullptr);
  const auto &args = impl_->Args();
  assert(args.axis == Dims - 2 || args.axis == Dims - 1);
  auto in_shape = in.shape;
  auto out_shape = out.shape;
  auto out_strides = GetStrides(out_shape);
  auto in_strides = GetStrides(in_shape);

  if (args.axis == Dims - 2) {
    auto nwin = in_shape[Dims - 1];
    ForAxis(out.data, in.data, out_shape.data(), out_strides.data(), in_shape.data(),
            in_strides.data(), Dims - 2,
            Dims - 1,  // Iterating slices of the two last dimensions
            [this, nwin](T *out_data, const T *in_data, int64_t out_size, int64_t out_stride,
                         int64_t in_size, int64_t in_stride) {
              impl_->ComputeFreqMajor(out_data, in_data, nwin);
            });
  } else {
    int64_t nwin = 1;
    for (int d = 0; d < Dims - 1; d++)
      nwin *= in_shape[d];
    impl_->ComputeTimeMajor(out.data, in.data, nwin);
  }
}

template class MelFilterBankCpu<float, 2>;
template class MelFilterBankCpu<double, 2>;

template class MelFilterBankCpu<float, 3>;
template class MelFilterBankCpu<double, 3>;

template class MelFilterBankCpu<float, 4>;
template class MelFilterBankCpu<double, 4>;

}  // namespace audio
}  // namespace kernels
}  // namespace dali
