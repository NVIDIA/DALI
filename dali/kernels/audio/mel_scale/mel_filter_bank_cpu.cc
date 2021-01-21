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
#include <utility>
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
template <typename T>
class MelFilterBankCpu<T>::Impl: public MelFilterImplBase<T> {
 public:
  template <typename MelScale>
  Impl(MelScale mel_scale, const MelFilterBankArgs &args)
      : MelFilterImplBase<T>(mel_scale, args) {
    auto nfilter = args.nfilter;

    intervals_.resize(fftbin_size_, -1);
    int fftbin = fftbin_start_;
    double f = fftbin * hz_step_;
    double mel = mel_low_ + mel_delta_;
    for (int interval = 0; interval < nfilter + 1; interval++, mel += mel_delta_) {
      double freq = mel_scale.mel_to_hz(interval == nfilter ? mel_high_ : mel);
      for (; fftbin <= fftbin_end_ && f < freq; fftbin++, f = fftbin * hz_step_) {
        intervals_[fftbin] = interval;
      }
    }

    mel = mel_low_ + mel_delta_;
    interval_ends_.resize(nfilter + 2);
    interval_ends_[0] = fftbin_start_;
    interval_ends_[nfilter + 1] = fftbin_end_ + 1;
    for (int interval = 1; interval < nfilter + 1; interval++, mel += mel_delta_) {
      double freq = mel_scale.mel_to_hz(mel);
      interval_ends_[interval] = std::ceil(freq / hz_step_);
    }
  }


  /**
   * @brief Applies mel filter bank to a 2D spectrogram, optimized for 
   *        frequency-major layout ("ft"). 
   */
  void ComputeFreqMajor(T* out, const T* in, int64_t nwindows,
                        int64_t out_size, int64_t out_stride,
                        int64_t in_size, int64_t in_stride) {
    for (int64_t m = 0; m < out_size; m++) {
      T* out_row = out + m * out_stride;
      for (int64_t t = 0; t < nwindows; t++)
        out_row[t] = T(0);
    }

    const T *in_row = in + fftbin_start_ * in_stride;
    for (int64_t fftbin = fftbin_start_; fftbin <= fftbin_end_; fftbin++, in_row += in_stride) {
      auto filter_up = intervals_[fftbin];
      auto weight_up = T(1) - weights_down_[fftbin];
      auto filter_down = filter_up - 1;
      auto weight_down = weights_down_[fftbin];

      if (filter_down >= 0) {
        if (args_.normalize)
          weight_down *= norm_factors_[filter_down];
        auto *out_row = out + filter_down * out_stride;
        for (int t = 0; t < nwindows; t++) {
          out_row[t] += weight_down * in_row[t];
        }
      }

      if (filter_up >= 0 && filter_up < out_size) {
        if (args_.normalize)
          weight_up *= norm_factors_[filter_up];
        auto *out_row = out + filter_up * out_stride;
        for (int t = 0; t < nwindows; t++) {
          out_row[t] += weight_up * in_row[t];
        }
      }
    }
  }

  /**
   * @brief Applies a mel filter bank to a one-dimensional spectrum input or
   *        individual frames in a time-major layout spectrogram input ("tf").
   */
  void ComputeTimeMajor(T* out, const T* in, int64_t nfilter, int64_t fftbin_size) {
    for (int m = 0; m < nfilter; m++) {
      T val = 0;
      int fftbin = interval_ends_[m];
      int f1 = interval_ends_[m + 1];
      int f2 = interval_ends_[m + 2];
      for (; fftbin < f1; ++fftbin) {
        auto weight_up = T(1) - weights_down_[fftbin];
        val += in[fftbin] * weight_up;
      }
      for (; fftbin < f2; ++fftbin) {
        val += in[fftbin] * weights_down_[fftbin];
      }
      if (args_.normalize)
        val *= norm_factors_[m];
      *out++ = val;
    }
  }

 private:
  std::vector<int> intervals_;
  std::vector<int> interval_ends_;
  USE_MEL_FILTER_IMPL_MEMBERS(T);
};

template <typename T>
MelFilterBankCpu<T>::MelFilterBankCpu() = default;

template <typename T>
MelFilterBankCpu<T>::~MelFilterBankCpu() = default;

template <typename T>
KernelRequirements MelFilterBankCpu<T>::Setup(KernelContext &context,
                                              const InTensorCPU<T> &in,
                                              const MelFilterBankArgs &orig_args) {
  auto args = orig_args;
  int ndim = in.dim();
  args.axis = args.axis >= 0 ? args.axis : ndim - 2;
  assert(args.axis >= 0 && args.axis <= ndim - 1);
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

template <typename T>
void MelFilterBankCpu<T>::Run(KernelContext &context,
                              const OutTensorCPU<T> &out,
                              const InTensorCPU<T> &in) {
  DALI_ENFORCE(impl_ != nullptr);
  const auto &args = impl_->Args();

  TensorShape<DynamicDimensions> in_shape = in.shape;
  TensorShape<DynamicDimensions> out_shape = out.shape;
  auto axis = args.axis;
  if (axis > 1) {
    in_shape = collapse_dims(in_shape, {std::make_pair(0, axis)});
    out_shape = collapse_dims(out_shape, {std::make_pair(0, axis)});
    axis = 1;
  }
  if (axis < in_shape.size() - 2) {
    in_shape = collapse_dims(in_shape, {std::make_pair(axis + 1, in_shape.size() - axis - 1)});
    out_shape = collapse_dims(out_shape, {std::make_pair(axis + 1, out_shape.size() - axis - 1)});
  }

  bool is_freq_last = axis == in_shape.size() - 1 || in_shape[in_shape.size() - 1] == 1;

  assert(in_shape.size() <= 3);
  int64_t fftbin_size = in_shape[axis];
  int64_t nfilter = out_shape[axis];

  if (is_freq_last) {
    int64_t nwindows = axis == 0 ? 1 : in_shape[0];
    for (int t = 0; t < nwindows; t++) {
      const T *in_row = in.data + t * fftbin_size;
      T *out_row = out.data + t * nfilter;
      impl_->ComputeTimeMajor(out_row, in_row, nfilter, fftbin_size);
    }
  } else {
    int64_t nwindows = 1;
    // Grouping last two dimensions as "ft"
    auto f_in_size = in_shape[axis];
    auto f_in_stride = GetStrides(in_shape)[axis];
    auto f_out_size = out_shape[axis];
    auto f_out_stride = GetStrides(out_shape)[axis];
    if (axis < in_shape.size() - 1) {
      nwindows = in_shape[in_shape.size() - 1];
      in_shape = collapse_dim(in_shape, axis);
      out_shape = collapse_dim(out_shape, axis);
    }
    auto in_strides = GetStrides(in_shape);
    auto out_strides = GetStrides(out_shape);

    ForAxis(out.data, in.data, out_shape.data(), out_strides.data(), in_shape.data(),
            in_strides.data(), axis, in_shape.size(),
            [this, nwindows, f_out_size, f_out_stride, f_in_size, f_in_stride]
            (T *out_data, const T *in_data, int64_t out_size, int64_t out_stride,
             int64_t in_size, int64_t in_stride) {
              impl_->ComputeFreqMajor(out_data, in_data, nwindows,
                                      f_out_size, f_out_stride, f_in_size, f_in_stride);
            });
  }
}

template class MelFilterBankCpu<float>;
template class MelFilterBankCpu<double>;

}  // namespace audio
}  // namespace kernels
}  // namespace dali
