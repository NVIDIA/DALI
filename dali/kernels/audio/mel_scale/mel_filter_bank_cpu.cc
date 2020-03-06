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
    intervals_.resize(fftbin_size_, -1);
    double mel = mel_low_ + mel_delta_;

    int64_t fftbin = fftbin_start_;
    double f = fftbin * hz_step_;

    int last_interval = args_.nfilter;
    for (int64_t interval = 0; interval <= last_interval; interval++, mel += mel_delta_) {
      if (interval == last_interval) {
        mel = mel_high_;
      }
      double freq = mel_scale.mel_to_hz(mel);
      for (; fftbin <= fftbin_end_ && f < freq; fftbin++, f = fftbin * hz_step_) {
        intervals_[fftbin] = interval;
      }
    }
  }

  void Compute(T* out, const T* in, int64_t nwindows,
               int64_t out_stride = -1, int64_t in_stride = -1) {
    if (out_stride <= 0)
      out_stride = nwindows;

    if (in_stride <= 0)
      in_stride = nwindows;

    int nfilter = args_.nfilter;

    std::memset(out, 0, sizeof(T) * nfilter * nwindows);
    for (int64_t fftbin = fftbin_start_; fftbin <= fftbin_end_; fftbin++) {
      auto *in_row_start = in + fftbin * in_stride;
      auto filter_up = intervals_[fftbin];
      auto weight_up = T(1) - weights_down_[fftbin];
      auto filter_down = filter_up - 1;
      auto weight_down = weights_down_[fftbin];

      if (filter_down >= 0) {
        if (args_.normalize)
          weight_down *= norm_factors_[filter_down];
        auto *out_row_start = out + filter_down * out_stride;
        for (int t = 0; t < nwindows; t++) {
          out_row_start[t] += weight_down * in_row_start[t];
        }
      }

      if (filter_up >= 0 && filter_up < nfilter) {
        if (args_.normalize)
          weight_up *= norm_factors_[filter_up];
        auto *out_row_start = out + filter_up * out_stride;
        for (int t = 0; t < nwindows; t++) {
          out_row_start[t] += weight_up * in_row_start[t];
        }
      }
    }
  }

 private:
  std::vector<int> intervals_;
  USE_MEL_FILTER_IMPL_MEMBERS(T, Dims);
};

template <typename T, int Dims>
MelFilterBankCpu<T, Dims>::MelFilterBankCpu() = default;

template <typename T, int Dims>
MelFilterBankCpu<T, Dims>::~MelFilterBankCpu() = default;

template <typename T, int Dims>
KernelRequirements MelFilterBankCpu<T, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<T, Dims> &in,
    const MelFilterBankArgs &original_args) {
  auto args = original_args;
  args.axis = args.axis >= 0 ? args.axis : Dims - 2;
  DALI_ENFORCE(args.axis == Dims - 2,
    "Input is expected to be a spectrogram with the last two dimensions being FFT bin index and "
    "window index respectively");
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
void MelFilterBankCpu<T, Dims>::Run(
    KernelContext &context,
    const OutTensorCPU<T, Dims> &out,
    const InTensorCPU<T, Dims> &in,
    const MelFilterBankArgs &original_args) {
  (void) original_args;
  DALI_ENFORCE(impl_ != nullptr);
  const auto &args = impl_->Args();
  auto in_shape = in.shape;
  auto nwin = in_shape[Dims - 1];
  auto in_strides = GetStrides(in_shape);
  auto out_shape = out.shape;
  auto out_strides = GetStrides(out_shape);
  auto for_axis_ndim = out.dim() - 1;  // squeeze last dim
  ForAxis(
    out.data, in.data, out_shape.data(), out_strides.data(), in_shape.data(), in_strides.data(),
    args.axis, for_axis_ndim,
    [this, nwin](
        T *out_data, const T *in_data,
        int64_t out_size, int64_t out_stride, int64_t in_size, int64_t in_stride) {
      impl_->Compute(out_data, in_data, nwin);
    });
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
