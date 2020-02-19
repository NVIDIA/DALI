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

const int kBlock = 32;

template <typename T>
struct BlockDesc {
  T *in_sample;
  T *out_sample;
  int64_t start_window;
  int64_t nwindows;
};

template <typename T>
__global__ void MelFilterBankKernel(const BlockDesc<T> *block_desc,
                                    const T *weights_down, const int64_t *interval_ends,
                                    bool normalize, const T *norm_factors,
                                    int64_t fftbin_first, int64_t fftbin_last,
                                    int64_t mel_bins) {
  auto block_id = blockIdx.x;
  T *in_sample = block_desc[block_id].in_sample;
  T *out_sample = block_desc[block_id].out_sample;
  int64_t mel_bin = blockIdx.y * kBlock + threadIdx.y;
  if (mel_bin >= mel_bins) return;
  int64_t window = block_desc[block_id].start_window + threadIdx.x;
  int64_t nwindows = block_desc[block_id].nwindows;
  if (window >= nwindows) return;
  T *out = out_sample + mel_bin * nwindows + window;
  *out = T(0);
  // filter up
  int64_t fftbin = (mel_bin > 0) ? interval_ends[mel_bin - 1] + 1 : fftbin_first;
  int64_t fftbin_end = interval_ends[mel_bin];
  T *in =  in_sample + fftbin * nwindows + window;
  for (; fftbin <= fftbin_end; ++fftbin, in += nwindows) {
    auto weight_up = T(1) - weights_down[fftbin];
    if (normalize) weight_up *= norm_factors[mel_bin];
    *out += *in * weight_up;
  }
  // filter down
  fftbin = interval_ends[mel_bin] + 1;
  fftbin_end = (mel_bin < mel_bins - 1) ? interval_ends[mel_bin + 1] : fftbin_last;
  *in =  in_sample + fftbin * nwindows + window;
  for (; fftbin <= fftbin_end; ++fftbin, in += nwindows) {
    auto weight_down = weights_down[fftbin];
    if (normalize) weight_down *= norm_factors[mel_bin + 1];
    *out += *in * weight_down;
  }
}




template <typename T, int Dims>
class MelFilterBankGpu<T, Dims>::Impl : MelFilterImplBase<T, Dims> {
 public:
  template <typename MelScale>
  Impl(MelScale mel_scale, const MelFilterBankArgs &args, ScratchpadEstimator &se,
       const TensorListShape<Dims> &in_shape)
      : MelFilterImplBase<T, Dims>(mel_scale, args) {
    T mel = mel_low_ + mel_delta_;
    for (int64_t interval = 0; interval < args_.nfilter; interval++, mel += mel_delta_) {
      T freq = mel_scale.mel_to_hz(mel);
      interval_ends_[interval] = std::floor(freq / hz_step_);
    }
    se.add<int64_t>(AllocType::GPU, interval_ends_.size());
    se.add<T>(AllocType::GPU, weights_down_.size());
    if (args_.normalize)
      se.add<T>(AllocType::GPU, norm_factors_.size());

    auto batch_size = in_shape.num_samples();
    for (int64_t ti = 0; ti < batch_size; ++ti) {
      const auto &tshape = in_shape.tensor_shape(ti);
      nsamples_.push_back(volume(tshape.data(), tshape.data() + args_.axis));
      nwindows_.push_back(volume(tshape.data() + args_.axis + 1, tshape.data() + tshape.size()));
      for (int64_t s = 0; s < nsamples_.back(); ++s) {
        auto nblocks = (nwindows_.back() + kBlock - 1) / kBlock;
        for (int64_t b = 0; b < nblocks; ++b) {
          block_descs_.push_back(BlockDesc<T>{nullptr, nullptr, b * kBlock, nwindows_.back()});
        }
      }
    }
    se.add<BlockDesc<T>>(AllocType::GPU, block_descs_.size());
  }

  void Compute(T** in_list, T **out_list, int64_t fft_dim,
               Scratchpad &scratchpad, cudaStream_t stream) {
    int64_t block_id = 0;
    for (int64_t ti = 0; ti < nsamples_.size(); ++ti) {
      T *in = in_list[ti];
      T *out = out_list[ti];
      for (int64_t s = 0; s < nsamples_[ti]; ++s) {
        block_descs_[block_id].in_sample = in;
        block_descs_[block_id].out_sample = out;
        in += nwindows_[ti] * fft_dim;
        out += nwindows_[ti] * args_.nfilter;
        ++block_id;
      }
    }
    auto block_descs = scratchpad.ToGPU(stream, block_descs_);
    auto interval_ends = scratchpad.ToGPU(stream, interval_ends_);
    auto weights_down = scratchpad.ToGPU(stream, weights_down_);
    T *norm_factors = nullptr;
    if (args_.normalize)
      norm_factors = scratchpad.ToGPU(stream, norm_factors_);
    dim3 block(kBlock, kBlock);
    dim3 grid(block_descs_.size(), (args_.nfilter + kBlock - 1) / kBlock);
    MelFilterBankKernel
      <<<grid, block, 0, stream>>>(block_descs, weights_down, interval_ends, args_.normalize,
                                   norm_factors, fftbin_start_, fftbin_end_, args_.nfilter)

  }

 private:
  std::vector<int64_t> interval_ends_;
  std::vector<int64_t> nsamples_;
  std::vector<int64_t> nwindows_;
  std::vector<BlockDesc<T>> block_descs_;
  USE_MEL_FILTER_IMPL_MEMBERS(T, Dims);
};

template <typename T, int Dims>
KernelRequirements MelFilterBankGpu<T, Dims>::Setup(KernelContext &context,
                                                    const InListGPU<T, Dims> &in,
                                                    const MelFilterBankArgs &original_args) {
  auto args = original_args;
  args.axis = args.axis >= 0 ? args.axis : Dims - 2;
  auto out_shape = in.shape;
  for (auto &tshape : out_shape) {
    tshape[args.axis] = args.nfilter;
  }

  KernelRequirements req;
  req.output_shapes = {out_shape};

  // koknstruktor impl - przekazac scratchpad estimator
  args.nfft = args.nfft > 0 ? args.nfft : 2 * (in.shape[args.axis] - 1);
  args.freq_high = args.freq_high > 0 ? args.freq_high : args.sample_rate / 2;
  if (!impl_ || impl_->Args() != args) {
    impl_.reset();
    switch (args.mel_formula) {
      case MelScaleFormula::HTK:
        impl_ = std::make_unique<Impl>(HtkMelScale<T>(), args); // tu
        break;
      case MelScaleFormula::Slaney:
      default:
        impl_ = std::make_unique<Impl>(SlaneyMelScale<T>(), args);
        break;
    }
  }
  return req;
}
}

}  // namespace audio
}  // namespace kernels
}